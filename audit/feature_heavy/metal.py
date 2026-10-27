"""Research Metal sorting, exact histograms, certified KSG neighbor proposals."""

from functools import lru_cache

import numpy as np

from fast_select._metal import get_mx

from .common import repair_proposals


@lru_cache(maxsize=1)
def sort_kernel():
    return get_mx().fast.metal_kernel(
        name="wide_exact_bitonic",
        input_names=["high", "low"],
        output_names=["orders"],
        source=r"""
        uint f = threadgroup_position_in_grid.x, lane = thread_index_in_threadgroup;
        uint n = high_shape[1];
        threadgroup uint kh[PAD], kl[PAD], ids[PAD];
        for (uint i=lane; i<PAD; i+=LANES) {
            kh[i] = i<n ? high[f*n+i] : 0xffffffffu;
            kl[i] = i<n ? low[f*n+i] : 0xffffffffu;
            ids[i] = i<n ? i : 0xffffffffu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint width=2; width<=PAD; width*=2) {
            for (uint gap=width/2; gap>0; gap/=2) {
                for (uint i=lane; i<PAD; i+=LANES) {
                    uint j=i^gap;
                    if (j>i) {
                        bool greater = kh[i]>kh[j] || (kh[i]==kh[j] &&
                            (kl[i]>kl[j] || (kl[i]==kl[j] && ids[i]>ids[j])));
                        bool less = kh[i]<kh[j] || (kh[i]==kh[j] &&
                            (kl[i]<kl[j] || (kl[i]==kl[j] && ids[i]<ids[j])));
                        bool swap = ((i&width)==0) ? greater : less;
                        if (swap) {
                            uint h=kh[i], l=kl[i], id=ids[i];
                            kh[i]=kh[j]; kl[i]=kl[j]; ids[i]=ids[j];
                            kh[j]=h; kl[j]=l; ids[j]=id;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint i=lane; i<n; i+=LANES) orders[f*n+i]=ids[i];
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def histogram_kernel():
    return get_mx().fast.metal_kernel(
        name="wide_ordered_histograms",
        input_names=["columns", "target", "active", "selection"],
        output_names=["pair", "triple"],
        source=r"""
        uint f=threadgroup_position_in_grid.x, lane=thread_index_in_threadgroup;
        uint n=columns_shape[1], selected=selection[0];
        threadgroup atomic_uint counts[T], first[T], pcounts[S*S], pfirst[S*S];
        for (uint i=lane;i<T;i+=128) {
            atomic_store_explicit(&counts[i],0u,memory_order_relaxed);
            atomic_store_explicit(&first[i],n,memory_order_relaxed);
            triple[f*T+i]=0;
        }
        for (uint i=lane;i<S*S;i+=128) {
            atomic_store_explicit(&pcounts[i],0u,memory_order_relaxed);
            atomic_store_explicit(&pfirst[i],n,memory_order_relaxed);
            pair[f*S*S+i]=0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active[f]) return;
        for (uint i=lane;i<n;i+=128) {
            uint key=uint(columns[selected*n+i])*S+uint(columns[f*n+i]);
            atomic_fetch_add_explicit(&pcounts[key],1u,memory_order_relaxed);
            atomic_fetch_min_explicit(&pfirst[key],i,memory_order_relaxed);
            if (COND) {
                key=key*SY+uint(target[i]);
                atomic_fetch_add_explicit(&counts[key],1u,memory_order_relaxed);
                atomic_fetch_min_explicit(&first[key],i,memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint key=lane;key<S*S;key+=128) {
            uint seen=atomic_load_explicit(&pfirst[key],memory_order_relaxed);
            if (seen<n) {
                uint rank=0;
                for (uint j=0;j<S*S;++j)
                    rank+=atomic_load_explicit(&pfirst[j],memory_order_relaxed)<seen;
                pair[f*S*S+rank]=atomic_load_explicit(&pcounts[key],memory_order_relaxed);
            }
        }
        if (COND) for (uint key=lane;key<T;key+=128) {
            uint seen=atomic_load_explicit(&first[key],memory_order_relaxed);
            if (seen<n) {
                uint rank=0;
                for (uint j=0;j<T;++j)
                    rank+=atomic_load_explicit(&first[j],memory_order_relaxed)<seen;
                triple[f*T+rank]=atomic_load_explicit(&counts[key],memory_order_relaxed);
            }
        }
        """,
        compile_options={"math_mode": "safe"},
    )


class Histograms:
    def __init__(self, columns, target, counts, sy):
        self.mx = get_mx()
        self.p, self.n = columns.shape
        self.s, self.sy = int(counts.max()), int(sy)
        if self.s > 8 or self.sy > 8:
            raise ValueError("Experimental histogram supports up to eight states/classes")
        with self.mx.stream(self.mx.gpu):
            self.columns = self.mx.array(columns)
            self.target = self.mx.array(target)
            self.mx.eval(self.columns, self.target)

    def step(self, selected, active, conditional):
        mx = self.mx
        with mx.stream(mx.gpu):
            values = histogram_kernel()(
                inputs=[self.columns, self.target, mx.array(active), mx.array([selected], dtype=mx.uint32)],
                template=[("S", self.s), ("SY", self.sy), ("T", self.s * self.s * self.sy), ("COND", int(conditional))],
                grid=(self.p * 128, 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(self.p, self.s * self.s), (self.p, self.s * self.s * self.sy)],
                output_dtypes=[mx.uint32, mx.uint32],
            )
            pair = np.array(values[0])
            triple = np.array(values[1]) if conditional else np.empty((self.p, 0), dtype=np.uint32)
            return pair, triple


@lru_cache(maxsize=1)
def proposal_kernel():
    return get_mx().fast.metal_kernel(
        name="wide_ksg_proposals",
        input_names=["columns", "target"],
        output_names=["neighbors", "cutoff"],
        source=r"""
        uint task=thread_position_in_grid.x;
        uint n=columns_shape[1], p=columns_shape[0];
        if (task>=p*n) return;
        uint f=task/n, i=task%n;
        float best[M]; uint ids[M];
        for (uint j=0;j<M;++j) {best[j]=INFINITY; ids[j]=n;}
        float x=columns[f*n+i], y=target[i];
        for (uint j=0;j<n;++j) if (j!=i) {
            float d=max(fabs(x-columns[f*n+j]),fabs(y-target[j]));
            if (d<best[0]) {
                uint pos=0;
                while(pos+1<M && best[pos+1]>d) {
                    best[pos]=best[pos+1]; ids[pos]=ids[pos+1]; ++pos;
                }
                best[pos]=d; ids[pos]=j;
            }
        }
        for (uint j=0;j<M;++j) neighbors[task*M+j]=ids[j];
        cutoff[task]=best[0];
        """,
        compile_options={"math_mode": "safe"},
    )


def sort(high, low):
    mx = get_mx()
    p, n = high.shape
    pad = 1 << (n - 1).bit_length()
    if pad > 1024:
        raise ValueError("Experimental sort supports up to 1024 samples")
    lanes = min(256, pad)
    with mx.stream(mx.gpu):
        output = sort_kernel()(
            inputs=[mx.array(high), mx.array(low)],
            template=[("PAD", pad), ("LANES", lanes)],
            grid=(p * lanes, 1, 1),
            threadgroup=(lanes, 1, 1),
            output_shapes=[(p, n)],
            output_dtypes=[mx.uint32],
        )[0]
        return np.array(output)


def histograms(columns, target, counts, sy):
    return Histograms(columns, target, counts, sy)


def radii(columns, target, k):
    if k > 16:
        raise ValueError("Experimental Metal proposals support k<=16")
    mx = get_mx()
    p, n = columns.shape
    high = columns.astype(np.float32)
    yt = target.astype(np.float32)
    if not np.isfinite(high).all() or not np.isfinite(yt).all():
        raise ValueError("Coordinates exceed experimental Metal proposal range")
    # Coordinate quantization + subtraction rounding (with ample slack).
    error_x = np.max(np.abs(columns - high.astype(np.float64)), axis=1)
    error_y = np.max(np.abs(target - yt.astype(np.float64)))
    magnitude = np.maximum(np.max(np.abs(columns), axis=1), np.max(np.abs(target)))
    error = 2 * np.maximum(error_x, error_y) + 16 * np.finfo(np.float32).eps * magnitude + 8 * np.finfo(np.float32).tiny
    with mx.stream(mx.gpu):
        ids, cut = proposal_kernel()(
            inputs=[mx.array(high), mx.array(yt)],
            template=[("M", k + 2)],
            grid=(p * n, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(p, n, k + 2), (p, n)],
            output_dtypes=[mx.uint32, mx.float32],
        )
        neighbors, cutoff = np.array(ids), np.array(cut).astype(np.float64)
    repaired, number = repair_proposals(columns, target, neighbors, cutoff, error, k)
    return repaired, {"float64_fallback_queries": int(number), "queries": p * n}
