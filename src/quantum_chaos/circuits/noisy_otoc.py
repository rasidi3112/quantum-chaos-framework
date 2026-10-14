from collections.abc import Callable, Sequence

import numpy as np
import pennylane as qml

from ..noise.noise_channels import NoiseModel, pennylane_depolarizing_parameter
from .trotter_evolution import TrotterEvolution

PAULI_GATES = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def richardson_extrapolate(scales: Sequence[float], values: Sequence[float]) -> float:
    x = np.asarray(scales, dtype=float)
    y = np.asarray(values, dtype=float)
    total = 0.0
    for i in range(x.size):
        weight = np.prod([xj / (xj - x[i]) for j, xj in enumerate(x) if j != i])
        total += weight * y[i]
    return float(total)


class NoisyTrotterOTOC:
    def __init__(
        self,
        pauli_terms: Sequence[tuple[float, str]],
        n_qubits: int,
        n_steps: int,
        noise: NoiseModel | None = None,
        order: int = 1,
    ):
        self.terms = [(c, s) for c, s in pauli_terms if any(p != "I" for p in s)]
        self.n = n_qubits
        self.n_steps = n_steps
        self.noise = noise
        self.order = order
        self._trotter = TrotterEvolution(n_qubits)
        self.ancilla = n_qubits
        self.reference = list(range(n_qubits + 1, 2 * n_qubits + 1))
        device = "default.mixed" if noise is not None else "default.qubit"
        self.device = qml.device(device, wires=2 * n_qubits + 1)

    def _noise_layer(self) -> None:
        if self.noise is None or self.noise.depolarizing_rate == 0:
            return
        q = pennylane_depolarizing_parameter(self.noise.depolarizing_rate)
        for w in range(self.n):
            qml.DepolarizingChannel(q, wires=w)

    def _layer(self, dt: float) -> None:
        if self.order == 1:
            self._trotter.first_order_trotter_layer(self.terms, dt)
        else:
            self._trotter.second_order_trotter_layer(self.terms, dt)

    def _inverse_layer(self, dt: float) -> None:
        if self.order == 1:
            for c, s in reversed(self.terms):
                self._trotter._pauli_rotation(s, -c * dt)
        else:
            self._trotter.second_order_trotter_layer(self.terms, -dt)

    def _evolve(self, dt: float, fold: int) -> None:
        for _ in range(self.n_steps):
            self._layer(dt)
            self._noise_layer()
            for _ in range((fold - 1) // 2):
                self._inverse_layer(dt)
                self._noise_layer()
                self._layer(dt)
                self._noise_layer()

    def _evolve_back(self, dt: float, fold: int) -> None:
        for _ in range(self.n_steps):
            self._inverse_layer(dt)
            self._noise_layer()
            for _ in range((fold - 1) // 2):
                self._layer(dt)
                self._noise_layer()
                self._inverse_layer(dt)
                self._noise_layer()

    def circuit(
        self, W: tuple[str, int] | None, V: tuple[str, int] | None, fold: int = 1
    ) -> Callable[[float], float]:
        if fold < 1 or fold % 2 == 0:
            raise ValueError("fold must be an odd positive integer")

        def controlled(op: tuple[str, int] | None) -> None:
            if op is not None:
                qml.ctrl(PAULI_GATES[op[0]](wires=op[1]), control=self.ancilla)

        @qml.qnode(self.device)
        def hadamard_test(t: float):
            dt = t / self.n_steps
            for r, q in zip(self.reference, range(self.n), strict=True):
                qml.Hadamard(wires=r)
                qml.CNOT(wires=[r, q])
            qml.Hadamard(wires=self.ancilla)
            controlled(V)
            self._evolve(dt, fold)
            controlled(W)
            self._evolve_back(dt, fold)
            controlled(V)
            self._evolve(dt, fold)
            controlled(W)
            self._evolve_back(dt, fold)
            qml.Hadamard(wires=self.ancilla)
            return qml.expval(qml.PauliZ(self.ancilla))

        def otoc(t: float) -> float:
            return float(hadamard_test(t))

        return otoc

    def otoc(self, W: tuple[str, int], V: tuple[str, int], fold: int = 1):
        return self.circuit(W, V, fold)

    def echo(self, V: tuple[str, int], fold: int = 1):
        return self.circuit(None, V, fold)


def zero_noise_extrapolated(
    measure: Callable[[int], float], folds: Sequence[int] = (1, 3, 5)
) -> dict[str, float | list[float]]:
    values = [measure(f) for f in folds]
    return {
        "folds": list(folds),
        "values": values,
        "extrapolated": richardson_extrapolate(folds, values),
    }
