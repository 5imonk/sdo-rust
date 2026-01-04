#!/usr/bin/env python3
"""
Scikit-learn-ähnliche API für SDOstream (Sparse Data Observers Streaming)

Diese Klasse bietet eine kompatible API zu scikit-learn's Outlier Detection
Algorithmen für Streaming-Daten, sodass SDOstream einfach in bestehende
Machine Learning Pipelines integriert werden kann.
"""

import numpy as np
from sdo import SDOstream, SDOstreamParams


class SDOstreamOutlierDetector:
    """
    Sparse Data Observers Streaming (SDOstream) Outlier Detection mit scikit-learn-ähnlicher API.
    
    Diese Klasse implementiert eine kompatible Schnittstelle zu scikit-learn's
    Outlier Detection Algorithmen für Streaming-Daten, sodass SDOstream einfach
    in bestehende Pipelines integriert werden kann.
    
    Parameters
    ----------
    k : int, default=10
        Anzahl der Observer (Modellgröße). Muss fest sein für Streaming.
    
    x : int, default=5
        Anzahl der nächsten Nachbarn für Observations.
    
    t : float, default=10.0
        T-Parameter für fading: f = exp(-T^-1). Größere Werte bedeuten
        langsamere Anpassung an neue Daten.
    
    distance : str, default="euclidean"
        Distanzmetrik: "euclidean", "manhattan", "chebyshev", "minkowski"
    
    minkowski_p : float, optional
        p-Parameter für Minkowski-Distanz (nur wenn distance="minkowski")
    
    tree_type : str, default="vptree"
        Baum-Typ: "vptree" (default) oder "kdtree"
    
    Attributes
    ----------
    sdostream_ : SDOstream
        Das interne SDOstream-Modell.
    
    n_features_in_ : int
        Anzahl der Features (Dimensionen) im Trainingsdatensatz.
    
    Examples
    --------
    >>> from sdostream_sklearn import SDOstreamOutlierDetector
    >>> import numpy as np
    >>> 
    >>> # Erstelle Detector
    >>> detector = SDOstreamOutlierDetector(k=10, x=5, t=10.0)
    >>> 
    >>> # Initialisiere mit Daten (optional)
    >>> X_init = np.array([[1, 2], [2, 3], [10, 11]], dtype=np.float64)
    >>> detector.fit(X_init)
    >>> 
    >>> # Streaming: Verarbeite einzelne Punkte
    >>> for point in new_streaming_data:
    >>>     score = detector.predict(point.reshape(1, -1))
    >>>     detector.partial_fit(point.reshape(1, -1))
    """
    
    def __init__(
        self,
        k=10,
        x=5,
        t=10.0,
        distance="euclidean",
        minkowski_p=None,
        tree_type="vptree",
    ):
        """
        Initialisiere den SDOstream Outlier Detector.
        
        Parameters
        ----------
        k : int, default=10
            Anzahl der Observer (Modellgröße).
        x : int, default=5
            Anzahl der nächsten Nachbarn für Observations.
        t : float, default=10.0
            T-Parameter für fading: f = exp(-T^-1).
        distance : str, default="euclidean"
            Distanzmetrik.
        minkowski_p : float, optional
            p-Parameter für Minkowski-Distanz.
        tree_type : str, default="vptree"
            Baum-Typ.
        """
        self.k = k
        self.x = x
        self.t = t
        self.distance = distance
        self.minkowski_p = minkowski_p
        self.tree_type = tree_type
        self.sdostream_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """
        Initialisiere das SDOstream-Modell mit Daten.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Initialisierungsdaten.
        y : Ignored
            Nicht verwendet, vorhanden für scikit-learn Kompatibilität.
        
        Returns
        -------
        self : object
            Gibt self zurück.
        """
        X = self._validate_input(X, fit=True)
        
        self.sdostream_ = SDOstream()
        params = SDOstreamParams(
            k=self.k,
            x=self.x,
            t=self.t,
            distance=self.distance,
            minkowski_p=self.minkowski_p,
            tree_type=self.tree_type,
        )
        self.sdostream_.initialize(X, params)
        
        return self
    
    def partial_fit(self, X, y=None):
        """
        Aktualisiere das Modell mit neuen Streaming-Daten.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Neue Datenpunkte (können mehrere sein, werden einzeln verarbeitet).
        y : Ignored
            Nicht verwendet, vorhanden für scikit-learn Kompatibilität.
        
        Returns
        -------
        self : object
            Gibt self zurück.
        """
        if self.sdostream_ is None:
            # Wenn noch nicht initialisiert, verwende fit()
            return self.fit(X, y)
        
        X = self._validate_input(X, fit=False)
        
        params = SDOstreamParams(
            k=self.k,
            x=self.x,
            t=self.t,
            distance=self.distance,
            minkowski_p=self.minkowski_p,
            tree_type=self.tree_type,
        )
        
        # Verarbeite jeden Punkt einzeln (Streaming)
        for point in X:
            point_2d = point.reshape(1, -1)
            self.sdostream_.learn(point_2d, params)
        
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
        if self.sdostream_ is None:
            raise ValueError("Modell muss zuerst mit fit() initialisiert werden.")
        
        X = self._validate_input(X, fit=False)
        
        scores = []
        for point in X:
            point_2d = point.reshape(1, -1)
            score = self.sdostream_.predict(point_2d)
            scores.append(score)
        
        return np.array(scores)
    
    def fit_predict(self, X, y=None):
        """
        Initialisiere das Modell und berechne Outlier-Scores für die Initialisierungsdaten.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Initialisierungsdaten.
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
    
    def n_observers(self):
        """
        Gibt die Anzahl der Observer zurück.
        
        Returns
        -------
        n_observers : int
            Anzahl der Observer im Modell.
        """
        if self.sdostream_ is None:
            raise ValueError("Modell muss zuerst mit fit() initialisiert werden.")
        return self.sdostream_.n_observers()
    
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
                raise ValueError("Modell muss zuerst mit fit() initialisiert werden")
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X hat {X.shape[1]} Features, aber Modell erwartet {self.n_features_in_} Features"
                )
        
        return X
    
    def __repr__(self):
        """String-Repräsentation des Objekts."""
        return (
            f"SDOstreamOutlierDetector(k={self.k}, x={self.x}, t={self.t:.2f}, "
            f"distance={self.distance})"
        )


# Beispiel-Verwendung
if __name__ == "__main__":
    print("=" * 60)
    print("SDOstream Scikit-learn-ähnliche API - Beispiel")
    print("=" * 60)
    
    # Erstelle Detector
    detector = SDOstreamOutlierDetector(k=10, x=5, t=10.0)
    print(f"\nDetector: {detector}")
    
    # Initialisiere mit Daten
    np.random.seed(42)
    init_data = np.random.randn(20, 2) * 1.5 + np.array([3.0, 3.0])
    
    print(f"\nInitialisiere mit {len(init_data)} Datenpunkten...")
    detector.fit(init_data)
    print(f"✓ Modell initialisiert mit {detector.n_observers()} Observern")
    
    # Streaming: Verarbeite neue Punkte
    print("\nStreaming-Verarbeitung:")
    streaming_points = [
        np.array([3.0, 3.0], dtype=np.float64),    # Normal
        np.array([15.0, 15.0], dtype=np.float64), # Outlier
        np.array([3.5, 3.5], dtype=np.float64),   # Normal
        np.array([20.0, 20.0], dtype=np.float64), # Outlier
    ]
    
    for i, point in enumerate(streaming_points, 1):
        point_2d = point.reshape(1, -1)
        
        # Vorher: Score berechnen
        score_before = detector.predict(point_2d)[0]
        
        # Punkt verarbeiten (Modell aktualisiert sich)
        detector.partial_fit(point_2d)
        
        # Nachher: Score berechnen
        score_after = detector.predict(point_2d)[0]
        
        print(f"  Punkt {i}: [{point[0]:5.1f}, {point[1]:5.1f}] "
              f"→ Score vorher: {score_before:.4f}, "
              f"nachher: {score_after:.4f}")
    
    print(f"\nFinale Anzahl Observer: {detector.n_observers()}")
    print("\n✓ Beispiel erfolgreich abgeschlossen!")

