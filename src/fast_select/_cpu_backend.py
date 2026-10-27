"""Backend validation for selectors whose GPU experiments failed admission."""


def resolve_cpu_backend(backend):
    if not isinstance(backend, str):
        raise ValueError("backend must be 'auto' or 'cpu'")
    name = backend.lower()
    if name in ("auto", "cpu"):
        return "cpu"
    if name in ("gpu", "metal"):
        raise ValueError("No GPU backend was retained for this selector; use backend='cpu' or 'auto'")
    raise ValueError("backend must be 'auto' or 'cpu'")
