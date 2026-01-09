#!/usr/bin/env python3
"""
Scikit-learn-ähnliche API für SDO (Sparse Data Observers)

Diese Klasse bietet eine kompatible API zu scikit-learn's Outlier Detection
Algorithmen, sodass SDO einfach in bestehende Machine Learning Pipelines
integriert werden kann.
"""

import numpy as np
from sdo import SDO, SDOParams


class SDOOutlierDetector:
    """
    Sparse Data Observers (SDO) Outlier Detection mit scikit-learn-ähnlicher API.
    
    Diese Klasse implementiert eine kompatible Schnittstelle zu scikit-learn's
    Outlier Detection Algorithmen, sodass SDO einfach in bestehende Pipelines
    integriert werden kann.
    
    Parameters
    ----------
    k : int, default=10
        Anzahl der zu samplenden Observer.
    
    x : int, default=5
        Anzahl der nächsten Nachbarn für Observations.
    
    rho : float, default=0.2
        Fraktion der Observer, die als inaktiv entfernt werden (0.0-1.0).
        Höhere Werte entfernen mehr Observer.
    
    Attributes
    ----------
    sdo_ : SDO
        Das interne SDO-Modell.
    
    n_features_in_ : int
        Anzahl der Features (Dimensionen) im Trainingsdatensatz.
    
    Examples
    --------
    >>> from sdo_sklearn import SDOOutlierDetector
    >>> import numpy as np
    >>> 
    >>> # Erstelle Detector
    >>> detector = SDOOutlierDetector(k=10, x=5, rho=0.2)
    >>> 
    >>> # Trainiere mit Daten
    >>> X = np.array([[1, 2], [2, 3], [10, 11]], dtype=np.float64)
    >>> detector.fit(X)
    >>> 
    >>> # Berechne Outlier-Scores
    >>> scores = detector.predict(X)
    >>> print(scores)
    """
    
    def __init__(self, k=10, x=5, rho=0.2):
        """
        Initialisiere den SDO Outlier Detector.
        
        Parameters
        ----------
        k : int, default=10
            Anzahl der zu samplenden Observer.
        x : int, default=5
            Anzahl der nächsten Nachbarn für Observations.
        rho : float, default=0.2
            Fraktion der Observer, die als inaktiv entfernt werden (0.0-1.0).
        """
        self.k = k
        self.x = x
        self.rho = rho
        self.sdo_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """
        Trainiere das SDO-Modell.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Trainingsdaten.
        y : Ignored
            Nicht verwendet, vorhanden für scikit-learn Kompatibilität.
        
        Returns
        -------
        self : object
            Gibt self zurück.
        """
        X = self._validate_input(X, fit=True)
        
        self.sdo_ = SDO()
        params = SDOParams(k=self.k, x=self.x, rho=self.rho)
        self.sdo_.initialize(params)
        self.sdo_.learn(X)
        
        return self
    
    def predict(self, X):
        """
        Berechne Outlier-Scores für die gegebenen Datenpunkte.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datenpunkte, für die Outlier-Scores berechnet werden sollen.
        
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier-Scores für jeden Datenpunkt. Höhere Werte bedeuten
            höhere Wahrscheinlichkeit, dass der Punkt ein Outlier ist.
        """
        if self.sdo_ is None:
            raise ValueError("Modell muss zuerst mit fit() trainiert werden.")
        
        X = self._validate_input(X, fit=False)
        
        scores = []
        for point in X:
            point_2d = point.reshape(1, -1)
            score = self.sdo_.predict(point_2d)
            scores.append(score)
        
        return np.array(scores)
    
    def fit_predict(self, X, y=None):
        """
        Trainiere das Modell und berechne Outlier-Scores für die Trainingsdaten.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Trainingsdaten.
        y : Ignored
            Nicht verwendet, vorhanden für scikit-learn Kompatibilität.
        
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier-Scores für jeden Datenpunkt.
        """
        return self.fit(X, y).predict(X)
    
    def score_samples(self, X):
        """
        Berechne Outlier-Scores (Alias für predict).
        
        Diese Methode ist kompatibel mit scikit-learn's Outlier Detection API.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datenpunkte.
        
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier-Scores für jeden Datenpunkt.
        """
        return self.predict(X)
    
    def decision_function(self, X):
        """
        Berechne Decision Function Werte (Alias für predict).
        
        Diese Methode ist kompatibel mit scikit-learn's Outlier Detection API.
        Negative Werte bedeuten Outlier, positive Werte normale Punkte.
        Hier geben wir einfach die Scores zurück (höhere = mehr Outlier).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datenpunkte.
        
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Decision Function Werte.
        """
        return self.predict(X)
    
    def get_active_observers(self):
        """
        Gibt die aktiven Observer als NumPy-Array zurück.
        
        Returns
        -------
        observers : ndarray of shape (n_observers, n_features)
            Die aktiven Observer.
        """
        if self.sdo_ is None:
            raise ValueError("Modell muss zuerst mit fit() trainiert werden.")
        
        # PyO3 übergibt Python automatisch, wenn die Methode von Python aus aufgerufen wird
        return self.sdo_.get_active_observers()
    
    def _validate_input(self, X, fit=False):
        """
        Validiere und konvertiere Eingabedaten.
        
        Parameters
        ----------
        X : array-like
            Eingabedaten.
        fit : bool
            Ob dies für fit() (True) oder predict() (False) ist.
        
        Returns
        -------
        X : ndarray
            Validierte und konvertierte Daten.
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError(f"X muss 1D oder 2D sein, hat aber {X.ndim} Dimensionen")
        
        if fit:
            if X.shape[0] == 0:
                raise ValueError("X darf nicht leer sein")
            self.n_features_in_ = X.shape[1]
        else:
            if self.n_features_in_ is None:
                raise ValueError("Modell muss zuerst mit fit() trainiert werden")
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X hat {X.shape[1]} Features, aber Modell erwartet {self.n_features_in_} Features"
                )
        
        return X
    
    def __repr__(self):
        """String-Repräsentation des Objekts."""
        return (f"SDOOutlierDetector(k={self.k}, x={self.x}, rho={self.rho})")


# Beispiel-Verwendung
if __name__ == "__main__":
    print("=" * 60)
    print("SDO Scikit-learn-ähnliche API - Beispiel")
    print("=" * 60)
    
    # Erstelle Detector
    detector = SDOOutlierDetector(k=10, x=5, rho=0.2)
    print(f"\nDetector: {detector}")
    
    # Generiere Beispiel-Daten
    np.random.seed(42)
    normal_data = np.random.randn(50, 2) * 1.5 + np.array([3.0, 3.0])
    outlier_data = np.array([
        [15.0, 15.0],
        [-5.0, -5.0],
        [20.0, 20.0],
    ])
    X = np.vstack([normal_data, outlier_data])
    
    print(f"\nTrainingsdaten: {X.shape[0]} Punkte, {X.shape[1]} Dimensionen")
    
    # Trainiere das Modell
    print("\nTrainiere Modell...")
    scores = detector.fit_predict(X)
    print("✓ Modell trainiert")
    
    # Zeige Top-Outlier
    print("\nTop 5 Outlier (höchste Scores):")
    top_indices = np.argsort(scores)[::-1][:5]
    for i, idx in enumerate(top_indices, 1):
        point = X[idx]
        score = scores[idx]
        is_outlier = idx >= len(normal_data)
        marker = "✓" if is_outlier else "✗"
        print(f"  {i}. {marker} Punkt [{point[0]:6.2f}, {point[1]:6.2f}]: Score = {score:.4f}")
    
    # Teste mit neuen Daten
    print("\nTeste mit neuen Daten:")
    new_points = np.array([
        [3.0, 3.0],    # Sollte normal sein
        [15.0, 15.0],  # Sollte Outlier sein
    ])
    new_scores = detector.predict(new_points)
    for point, score in zip(new_points, new_scores):
        print(f"  Punkt [{point[0]:5.1f}, {point[1]:5.1f}]: Score = {score:.4f}")
    
    print("\n✓ Beispiel erfolgreich abgeschlossen!")

