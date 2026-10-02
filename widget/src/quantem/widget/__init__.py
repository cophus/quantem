from importlib.metadata import PackageNotFoundError, version

from quantem.widget.diffsim import DiffractionSim

try:
    __version__ = version("quantem.widget")
except PackageNotFoundError:
    # Source-tree imports (e.g. `PYTHONPATH=src pytest`) skip pip install.
    __version__ = "0.0.0+local"

__all__ = ["DiffractionSim"]
