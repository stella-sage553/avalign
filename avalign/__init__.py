"""avalign: a toolkit for contrastive audio-visual alignment.

The package is split into a dependency-light NumPy core (preprocessing,
losses, metrics, data handling) and an optional PyTorch layer under
:mod:`avalign.torchext` for training neural encoders.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
