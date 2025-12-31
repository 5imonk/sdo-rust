#!/usr/bin/env python3
"""
Scikit-learn-ähnliche API für SDOclust (Sparse Data Observers Clustering)

Diese Klasse bietet eine kompatible API zu scikit-learn's Clustering
Algorithmen, sodass SDOclust einfach in bestehende Machine Learning Pipelines
integriert werden kann.
"""

import numpy as np
from sdo import SDOclust, SDOclustParams


class SDOclustClusterer:
    """
    Sparse Data Observers Clustering (SDOclust) mit scikit-learn-ähnlicher API.
    
    Diese Klasse implementiert eine kompatible Schnittstelle zu scikit-learn's
    Clustering Algorithmen, sodass SDOclust einfach in bestehende Pipelines
    integriert werden kann.
    
    Parameters
    ----------
    k : int, default=10
        Anzahl der zu samplenden Observer.
    
    x : int, default=5
        Anzahl der nächsten Nachbarn für Observations und Predictions.
    
    rho : float, default=0.2
        Fraktion der Observer, die als inaktiv entfernt werden (0.0-1.0).
    
    chi : int, default=4
        Anzahl der nächsten Observer für lokale Cutoff-Thresholds (χ).
    
    zeta : float, default=0.5
        Mixing-Parameter für globale/lokale Thresholds (0.0-1.0).
        Höhere Werte betonen lokale Thresholds mehr.
    
    min_cluster_size : int, default=2
        Minimale Clustergröße (e). Cluster mit weniger Observer werden entfernt.
    
    Attributes
    ----------
    sdoclust_ : SDOclust
        Das interne SDOclust-Modell.
    
    n_features_in_ : int
        Anzahl der Features (Dimensionen) im Trainingsdatensatz.
    
    labels_ : ndarray of shape (n_samples,)
        Cluster-Labels für die Trainingsdaten.
    
    n_clusters_ : int
        Anzahl der gefundenen Cluster.
    
    Examples
    --------
    >>> from sdoclust_sklearn import SDOclustClusterer
    >>> import numpy as np
    >>> 
    >>> # Erstelle Clusterer
    >>> clusterer = SDOclustClusterer(k=10, x=5, chi=4, zeta=0.5)
    >>> 
    >>> # Trainiere mit Daten
    >>> X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]], dtype=np.float64)
    >>> clusterer.fit(X)
    >>> 
    >>> # Vorhersage für neue Daten
    >>> labels = clusterer.predict(X)
    >>> print(labels)
    """
    
    def __init__(
        self,
        k=10,
        x=5,
        rho=0.2,
        chi=4,
        zeta=0.5,
        min_cluster_size=2,
    ):
        """
        Initialisiere den SDOclust Clusterer.
        
        Parameters
        ----------
        k : int, default=10
            Anzahl der zu samplenden Observer.
        x : int, default=5
            Anzahl der nächsten Nachbarn.
        rho : float, default=0.2
            Fraktion der Observer, die entfernt werden.
        chi : int, default=4
            Anzahl der nächsten Observer für lokale Thresholds.
        zeta : float, default=0.5
            Mixing-Parameter für globale/lokale Thresholds.
        min_cluster_size : int, default=2
            Minimale Clustergröße.
        """
        self.k = k
        self.x = x
        self.rho = rho
        self.chi = chi
        self.zeta = zeta
        self.min_cluster_size = min_cluster_size
        self.sdoclust_ = None
        self.n_features_in_ = None
        self.labels_ = None
        self.n_clusters_ = None
    
    def fit(self, X, y=None):
        """
        Trainiere das SDOclust-Modell.
        
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
        
        self.sdoclust_ = SDOclust()
        params = SDOclustParams(
            k=self.k,
            x=self.x,
            rho=self.rho,
            chi=self.chi,
            zeta=self.zeta,
            min_cluster_size=self.min_cluster_size,
        )
        self.sdoclust_.learn(X, params)
        
        # Berechne Labels für Trainingsdaten
        self.labels_ = self._predict_batch(X)
        self.n_clusters_ = self.sdoclust_.n_clusters()
        
        return self
    
    def predict(self, X):
        """
        Vorhersage von Cluster-Labels für die gegebenen Datenpunkte.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datenpunkte, für die Cluster-Labels vorhergesagt werden sollen.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster-Labels für jeden Datenpunkt. -1 bedeutet Outlier/kein Cluster.
        """
        if self.sdoclust_ is None:
            raise ValueError("Modell muss zuerst mit fit() trainiert werden.")
        
        X = self._validate_input(X, fit=False)
        return self._predict_batch(X)
    
    def fit_predict(self, X, y=None):
        """
        Trainiere das Modell und berechne Cluster-Labels für die Trainingsdaten.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Trainingsdaten.
        y : Ignored
            Nicht verwendet, vorhanden für scikit-learn Kompatibilität.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster-Labels für jeden Datenpunkt.
        """
        return self.fit(X, y).labels_
    
    def _predict_batch(self, X):
        """Hilfsfunktion für Batch-Prediction."""
        labels = []
        for point in X:
            point_2d = point.reshape(1, -1)
            label = self.sdoclust_.predict(point_2d)
            labels.append(label)
        return np.array(labels)
    
    def get_observer_labels(self):
        """
        Gibt die Labels der Observer zurück.
        
        Returns
        -------
        labels : list of int
            Labels der Observer. -1 bedeutet entfernt/kein Cluster.
        """
        if self.sdoclust_ is None:
            raise ValueError("Modell muss zuerst mit fit() trainiert werden.")
        return self.sdoclust_.get_observer_labels()
    
    def get_active_observers(self):
        """
        Gibt die aktiven Observer als NumPy-Array zurück.
        
        Returns
        -------
        observers : ndarray of shape (n_observers, n_features)
            Die aktiven Observer.
        """
        if self.sdoclust_ is None:
            raise ValueError("Modell muss zuerst mit fit() trainiert werden.")
        return self.sdoclust_.get_active_observers()
    
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
        return (
            f"SDOclustClusterer(k={self.k}, x={self.x}, rho={self.rho}, "
            f"chi={self.chi}, zeta={self.zeta}, min_cluster_size={self.min_cluster_size})"
        )


# Beispiel-Verwendung
if __name__ == "__main__":
    print("=" * 60)
    print("SDOclust Scikit-learn-ähnliche API - Beispiel")
    print("=" * 60)
    
    # Erstelle Clusterer
    clusterer = SDOclustClusterer(k=20, x=5, chi=4, zeta=0.5, min_cluster_size=2)
    print(f"\nClusterer: {clusterer}")
    
    # Generiere Beispiel-Daten mit zwei Clustern
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(30, 2) * 0.5 + np.array([8.0, 8.0])
    X = np.vstack([cluster1, cluster2])
    
    print(f"\nTrainingsdaten: {X.shape[0]} Punkte, {X.shape[1]} Dimensionen")
    print(f"  - Cluster 1: 30 Punkte um [2, 2]")
    print(f"  - Cluster 2: 30 Punkte um [8, 8]")
    
    # Trainiere das Modell
    print("\nTrainiere Modell...")
    labels = clusterer.fit_predict(X)
    print(f"✓ Modell trainiert")
    print(f"  - Gefundene Cluster: {clusterer.n_clusters_}")
    
    # Zeige Cluster-Verteilung
    print("\nCluster-Verteilung:")
    unique_labels = np.unique(labels[labels >= 0])
    for label in unique_labels:
        count = np.sum(labels == label)
        percentage = 100.0 * count / len(labels)
        print(f"  Cluster {label}: {count} Punkte ({percentage:.1f}%)")
    
    outlier_count = np.sum(labels < 0)
    if outlier_count > 0:
        print(f"  Outlier: {outlier_count} Punkte")
    
    # Teste mit neuen Daten
    print("\nTeste mit neuen Daten:")
    new_points = np.array([
        [2.0, 2.0],    # Sollte zu Cluster 1 gehören
        [8.0, 8.0],    # Sollte zu Cluster 2 gehören
        [5.0, 5.0],    # Zwischen den Clustern
        [15.0, 15.0],  # Outlier
    ])
    new_labels = clusterer.predict(new_points)
    for point, label in zip(new_points, new_labels):
        label_str = f"Cluster {label}" if label >= 0 else "Outlier"
        print(f"  Punkt [{point[0]:5.1f}, {point[1]:5.1f}]: {label_str}")
    
    print("\n✓ Beispiel erfolgreich abgeschlossen!")

