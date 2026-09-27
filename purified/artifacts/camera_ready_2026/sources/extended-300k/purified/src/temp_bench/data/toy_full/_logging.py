"""Structured [tag] logging following docs/aniket/logging_format.md conventions."""


def log(tag: str, msg: str, **kwargs: object) -> None:
    """Print a tagged log message with optional key-value pairs.

    Format: [tag] msg | k1=v1 | k2=v2

    Args:
        tag: Log tag (info, data, train, eval, done, error, result, summary).
        msg: Main message.
        **kwargs: Additional key-value pairs to append.
    """
    parts = [f"[{tag}] {msg}"]
    for k, v in kwargs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3e}")
        else:
            parts.append(f"{k}={v}")
    print(" | ".join(parts))


def log_sweep(idx: int, total: int, **kwargs: object) -> None:
    """Print a sweep progress message.

    Format: [sweep 03/18] k1=v1 | k2=v2

    Args:
        idx: Current sweep index (1-based).
        total: Total number of sweep configs.
        **kwargs: Key-value pairs for the sweep config.
    """
    width = len(str(total))
    parts = [f"[sweep {idx:0{width}d}/{total}]"]
    kv_parts = [f"{k}={v}" for k, v in kwargs.items()]
    if kv_parts:
        parts.append(" | ".join(kv_parts))
    print(" ".join(parts))
