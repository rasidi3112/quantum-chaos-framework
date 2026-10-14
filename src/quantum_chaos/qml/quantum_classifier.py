

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from .quantum_kernel import QuantumKernel, SYKKernel, HubbardKernel


class QuantumClassifier:
    def __init__(
        self,
        kernel: QuantumKernel,
        C: float = 1.0,
        scale_data: bool = True
    ):
        self.kernel = kernel
        self.C = C
        self.scale_data = scale_data
        
        self.scaler = StandardScaler() if scale_data else None
        self.classifier = None
        self.X_train = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumClassifier':
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
        self.classifier = SVC(kernel='precomputed', C=self.C)
        self.classifier.fit(K_train, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
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
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        y_pred = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }


class QuantumKernelClassificationBenchmark:
    def __init__(
        self,
        n_qubits: int = 4,
        evolution_time: float = 1.0
    ):
        self.n_qubits = n_qubits
        self.evolution_time = evolution_time
        
        # Initialize kernels
        self.syk_kernel = None
        self.hubbard_kernel = None
        self.syk_classifier = None
        self.hubbard_classifier = None
    
    def setup_classifiers(self):
        # SYK kernel (chaotic)
        self.syk_kernel = SYKKernel(
            n_qubits=self.n_qubits,
            coupling_strength=1.0,
            evolution_time=self.evolution_time,
            seed=42
        )
        self.syk_classifier = QuantumClassifier(self.syk_kernel)
        
        # Hubbard kernel (regular) - use n_qubits/2 sites for matching dimension
        n_sites = max(2, self.n_qubits // 2)
        self.hubbard_kernel = HubbardKernel(
            n_sites=n_sites,
            hopping_amplitude=1.0,
            onsite_interaction=1.0,
            evolution_time=self.evolution_time
        )
        self.hubbard_classifier = QuantumClassifier(self.hubbard_kernel)
    
    def generate_classification_data(
        self,
        n_samples: int = 100,
        n_features: int = 4,
        n_classes: int = 2,
        separation: float = 1.0,
        noise: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(random_state)
        
        samples_per_class = n_samples // n_classes
        
        X = []
        y = []
        
        for c in range(n_classes):
            # Class center
            center = separation * np.random.randn(n_features)
            
            # Generate samples around center
            class_samples = center + noise * np.random.randn(samples_per_class, n_features)
            
            X.append(class_samples)
            y.extend([c] * samples_per_class)
        
        X = np.vstack(X)
        y = np.array(y)
        
        # Split
        return train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    def run_benchmark(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        if self.syk_classifier is None:
            self.setup_classifiers()
        
        results = {}
        
        # Train and evaluate SYK classifier
        print("Training SYK-based classifier...")
        self.syk_classifier.fit(X_train, y_train)
        syk_eval = self.syk_classifier.evaluate(X_test, y_test)
        results['syk'] = {
            'accuracy': syk_eval['accuracy'],
            'classification_report': syk_eval['classification_report']
        }
        
        # Train and evaluate Hubbard classifier  
        print("Training Hubbard-based classifier...")
        self.hubbard_classifier.fit(X_train, y_train)
        hubbard_eval = self.hubbard_classifier.evaluate(X_test, y_test)
        results['hubbard'] = {
            'accuracy': hubbard_eval['accuracy'],
            'classification_report': hubbard_eval['classification_report']
        }
        
        # Compute kernel expressivity metrics
        print("Computing expressivity metrics...")
        syk_expressivity = self.syk_kernel.expressivity_metric(n_samples=min(50, len(X_train)))
        hubbard_expressivity = self.hubbard_kernel.expressivity_metric(n_samples=min(50, len(X_train)))
        
        results['expressivity'] = {
            'syk': syk_expressivity,
            'hubbard': hubbard_expressivity
        }
        
        # Comparison summary
        results['comparison'] = {
            'syk_wins': syk_eval['accuracy'] > hubbard_eval['accuracy'],
            'accuracy_difference': syk_eval['accuracy'] - hubbard_eval['accuracy'],
            'chaos_advantage': (
                syk_eval['accuracy'] > hubbard_eval['accuracy'] and 
                syk_expressivity['expressivity_score'] > hubbard_expressivity['expressivity_score']
            )
        }
        
        return results
    
    def run_full_comparison(
        self,
        n_trials: int = 5,
        n_samples: int = 100
    ) -> Dict:
        if self.syk_classifier is None:
            self.setup_classifiers()
        
        syk_accuracies = []
        hubbard_accuracies = []
        
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}")
            
            # Generate new data for each trial
            X_train, X_test, y_train, y_test = self.generate_classification_data(
                n_samples=n_samples,
                n_features=min(4, self.n_qubits),
                random_state=trial
            )
            
            # Run benchmark
            results = self.run_benchmark(X_train, y_train, X_test, y_test)
            
            syk_accuracies.append(results['syk']['accuracy'])
            hubbard_accuracies.append(results['hubbard']['accuracy'])
        
        return {
            'n_trials': n_trials,
            'syk_accuracies': syk_accuracies,
            'hubbard_accuracies': hubbard_accuracies,
            'syk_mean': np.mean(syk_accuracies),
            'syk_std': np.std(syk_accuracies),
            'hubbard_mean': np.mean(hubbard_accuracies),
            'hubbard_std': np.std(hubbard_accuracies),
            'syk_wins_count': sum(1 for s, h in zip(syk_accuracies, hubbard_accuracies) if s > h),
            'conclusion': self._draw_conclusion(syk_accuracies, hubbard_accuracies)
        }
    
    def _draw_conclusion(
        self,
        syk_accs: List[float],
        hubbard_accs: List[float]
    ) -> str:
        syk_mean = np.mean(syk_accs)
        hubbard_mean = np.mean(hubbard_accs)
        
        if syk_mean > hubbard_mean + 0.05:
            return "SYK (chaotic) kernel shows significant advantage"
        elif hubbard_mean > syk_mean + 0.05:
            return "Hubbard (regular) kernel shows significant advantage"
        else:
            return "No significant difference between chaos and regular dynamics"


def generate_quantum_embedded_features(
    kernel: QuantumKernel,
    X: np.ndarray,
    n_components: int = 10
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
