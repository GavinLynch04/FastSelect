"""Lazy optional MLX runtime shared by Metal backends."""

from functools import lru_cache, wraps


@lru_cache(maxsize=1)
def get_mx():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Metal requires MLX and an accessible Apple Silicon GPU; install fast-select[metal]."
        ) from exc
    if not mx.metal.is_available():
        raise RuntimeError("No usable Metal GPU is available")
    return mx


def metal_entry(function):
    """Honor explicit Metal use even if an application defaults MLX to CPU."""

    @wraps(function)
    def call(*args, **kwargs):
        mx = get_mx()
        with mx.stream(mx.gpu):
            return function(*args, **kwargs)

    return call
