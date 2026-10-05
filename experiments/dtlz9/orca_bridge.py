"""Check that the locally installed ORCA package is available."""
try:
    import orca  # noqa: F401
except ImportError as exc:
    raise ImportError("Install ORCA from the repository root: python -m pip install -e .") from exc
