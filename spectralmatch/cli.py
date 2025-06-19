import fire
import spectralmatch
import inspect
from importlib.metadata import version as get_version, PackageNotFoundError
import sys

def cli_version():
    try:
        print(get_version("spectralmatch"))
    except PackageNotFoundError:
        print("Spectralmatch (version unknown)")

def build_cli():
    class CLI:
        """"""

    for name in spectralmatch.__all__:
        func = getattr(spectralmatch, name, None)
        if callable(func):
            func.__doc__ = inspect.getdoc(func) or "No description available."
            setattr(CLI, name, staticmethod(func))

    return CLI

def main():
    if "--version" in sys.argv:
        cli_version()
        return
    fire.Fire(build_cli())