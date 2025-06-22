import fire
import spectralmatch
import inspect
from importlib.metadata import version as get_version, PackageNotFoundError
import sys


def _cli_version():
    try:
        print(get_version("spectralmatch"))
    except PackageNotFoundError:
        print("Spectralmatch (version unknown)")


def _build_cli():
    class CLI:
        """"""

    for name in spectralmatch.__all__:
        func = getattr(spectralmatch, name, None)
        if callable(func):
            func.__doc__ = inspect.getdoc(func) or "No description available."
            setattr(CLI, name, staticmethod(func))

    return CLI


def main():
    # Ensure print() is flushed immediately (line-buffered)
    sys.stdout.reconfigure(line_buffering=True)

    if "--version" in sys.argv:
        _cli_version()
        return
    fire.Fire(_build_cli())
