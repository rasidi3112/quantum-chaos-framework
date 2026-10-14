from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quantum-chaos")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

_LAZY = {
    "SYKHamiltonian": ".hamiltonians",
    "DrivenHubbardHamiltonian": ".hamiltonians",
    "TrotterEvolution": ".circuits",
    "OTOCCalculator": ".circuits",
    "NISQSimulator": ".noise",
    "NoiseChannels": ".noise",
    "QuantumKernel": ".qml",
    "QuantumClassifier": ".qml",
    "QuantumVisualizer": ".visualization",
    "jordan_wigner_transform": ".utils",
    "pauli_operators": ".utils",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
