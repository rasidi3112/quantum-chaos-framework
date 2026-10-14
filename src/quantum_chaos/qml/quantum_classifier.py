import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .quantum_kernel import HubbardKernel, QuantumKernel, SYKKernel


class QuantumClassifier:
    def __init__(self, kernel: QuantumKernel, C: float = 1.0, scale_data: bool = True):
        self.kernel = kernel
        self.C = C
        self.scale_data = scale_data

        self.scaler = StandardScaler() if scale_data else None
        self.classifier: SVC | None = None
        self.X_train: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumClassifier":
        # Preprocess
        if self.scaler:
            X = self.scaler.fit_transform(X)

        self.X_train = X

        # Compute quantum kernel matrix
        K_train = self.kernel.kernel_matrix(X)

        # Ensure positive semi-definiteness (numerical stability)
        K_train = (K_train + K_train.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(K_train))
        if min_eig < 0:
            K_train += (-min_eig + 1e-8) * np.eye(len(K_train))

        # Train SVM with precomputed kernel
        classifier = SVC(kernel="precomputed", C=self.C)
        classifier.fit(K_train, y)
        self.classifier = classifier

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.classifier is None or self.X_train is None:
            raise RuntimeError("Classifier must be fitted before prediction")

        # Preprocess
        if self.scaler:
            X = self.scaler.transform(X)

        # Compute kernel between test and training data
        K_test = self.kernel.kernel_matrix(X, self.X_train)

        return self.classifier.predict(K_test)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "predictions": y_pred,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }


class QuantumKernelClassificationBenchmark:
    def __init__(self, n_qubits: int = 4, evolution_time: float = 1.0):
        self.n_qubits = n_qubits
        self.evolution_time = evolution_time

        # Initialised lazily by setup_classifiers()
        self.syk_kernel: SYKKernel | None = None
        self.hubbard_kernel: HubbardKernel | None = None
        self.syk_classifier: QuantumClassifier | None = None
        self.hubbard_classifier: QuantumClassifier | None = None

    def setup_classifiers(
        self,
    ) -> tuple[SYKKernel, QuantumClassifier, HubbardKernel, QuantumClassifier]:
        """Create (once) the SYK and Hubbard kernels and their SVM classifiers."""
        if (
            self.syk_kernel is None
            or self.syk_classifier is None
            or self.hubbard_kernel is None
            or self.hubbard_classifier is None
        ):
            self.syk_kernel = SYKKernel(
                n_qubits=self.n_qubits,
                coupling_strength=1.0,
                evolution_time=self.evolution_time,
                seed=42,
            )
            self.syk_classifier = QuantumClassifier(self.syk_kernel)
            # n_qubits/2 sites so that both feature maps act on n_qubits qubits
            self.hubbard_kernel = HubbardKernel(
                n_sites=max(2, self.n_qubits // 2),
                hopping_amplitude=1.0,
                onsite_interaction=1.0,
                evolution_time=self.evolution_time,
            )
            self.hubbard_classifier = QuantumClassifier(self.hubbard_kernel)
        return self.syk_kernel, self.syk_classifier, self.hubbard_kernel, self.hubbard_classifier

    def generate_classification_data(
        self,
        n_samples: int = 100,
        n_features: int = 4,
        n_classes: int = 2,
        separation: float = 1.0,
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)

        samples_per_class = n_samples // n_classes

        X_parts = []
        y_parts: list[int] = []

        for c in range(n_classes):
            # Class center
            center = separation * rng.standard_normal(n_features)

            # Generate samples around center
            class_samples = center + noise * rng.standard_normal((samples_per_class, n_features))

            X_parts.append(class_samples)
            y_parts.extend([c] * samples_per_class)

        X = np.vstack(X_parts)
        y = np.array(y_parts)

        # Split
        return train_test_split(X, y, test_size=0.3, random_state=random_state)

    def run_benchmark(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """Train and test both kernel SVMs on one split (descriptive; no significance test)."""
        syk_kernel, syk_classifier, hubbard_kernel, hubbard_classifier = self.setup_classifiers()

        syk_eval = syk_classifier.fit(X_train, y_train).evaluate(X_test, y_test)
        hubbard_eval = hubbard_classifier.fit(X_train, y_train).evaluate(X_test, y_test)
        n_stats = min(50, len(X_train))
        return {
            "syk": {
                "accuracy": syk_eval["accuracy"],
                "classification_report": syk_eval["classification_report"],
            },
            "hubbard": {
                "accuracy": hubbard_eval["accuracy"],
                "classification_report": hubbard_eval["classification_report"],
            },
            "expressivity": {
                "syk": syk_kernel.expressivity_metric(n_samples=n_stats, rng=0),
                "hubbard": hubbard_kernel.expressivity_metric(n_samples=n_stats, rng=0),
            },
            # Single-split difference; not a significance statement (see PLAN 3E)
            "accuracy_difference": syk_eval["accuracy"] - hubbard_eval["accuracy"],
        }

    def run_full_comparison(self, n_trials: int = 5, n_samples: int = 100) -> dict:
        syk_accuracies = []
        hubbard_accuracies = []

        for trial in range(n_trials):
            # Generate new data for each trial
            X_train, X_test, y_train, y_test = self.generate_classification_data(
                n_samples=n_samples, n_features=min(4, self.n_qubits), random_state=trial
            )

            # Run benchmark
            results = self.run_benchmark(X_train, y_train, X_test, y_test)

            syk_accuracies.append(results["syk"]["accuracy"])
            hubbard_accuracies.append(results["hubbard"]["accuracy"])

        return {
            "n_trials": n_trials,
            "syk_accuracies": syk_accuracies,
            "hubbard_accuracies": hubbard_accuracies,
            "syk_mean": np.mean(syk_accuracies),
            "syk_std": np.std(syk_accuracies),
            "hubbard_mean": np.mean(hubbard_accuracies),
            "hubbard_std": np.std(hubbard_accuracies),
        }


def generate_quantum_embedded_features(
    kernel: QuantumKernel, X: np.ndarray, n_components: int = 10
) -> np.ndarray:
    # Compute kernel matrix
    K = kernel.kernel_matrix(X)

    # Center the kernel matrix
    n = len(K)
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

    # Select top components (sorted in ascending order, so take from end)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute embedded features
    features = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))

    return features
