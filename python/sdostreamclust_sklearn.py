#!/usr/bin/env python3
"""
Scikit-learn-ähnliche API für SDOstreamclust (Sparse Data Observers Streaming Clustering)

Diese Klasse bietet eine kompatible API zu scikit-learn's Clustering
Algorithmen für Streaming-Daten, sodass SDOstreamclust einfach in bestehende
Machine Learning Pipelines integriert werden kann.
"""

import numpy as np
from sdo import SDOstreamclust


class SDOstreamclustClusterer:
    """
    Sparse Data Observers Streaming Clustering (SDOstreamclust) mit scikit-learn-ähnlicher API.
    
    Diese Klasse implementiert eine kompatible Schnittstelle zu scikit-learn's
    Clustering Algorithmen für Streaming-Daten, sodass SDOstreamclust einfach
    in bestehende Pipelines integriert werden kann.
    
    Parameters
    ----------
    k : int, default=10
        Anzahl der Observer (Modellgröße). Muss fest sein für Streaming.
    
    x : int, default=5
        Anzahl der nächsten Nachbarn für Observations und Predictions.
    
    t_fading : float, default=10.0
        T-Parameter für fading: f = exp(-T_fading^-1). Größere Werte bedeuten
        langsamere Anpassung an neue Daten.
    

    
    chi : int, default=4
        Anzahl der nächsten Observer für lokale Cutoff-Thresholds (χ).
    
    zeta : float, default=0.5
        Mixing-Parameter für globale/lokale Thresholds (0.0-1.0).
        Höhere Werte betonen lokale Thresholds mehr.
    
    min_cluster_size : int, default=2
        Minimale Clustergröße (e). Cluster mit weniger Observer werden entfernt.
    
    distance : str, default="euclidean"
        Distanzmetrik: "euclidean", "manhattan", "chebyshev", "minkowski"
    
    minkowski_p : float, optional
        p-Parameter für Minkowski-Distanz (nur wenn distance="minkowski")
    
    Attributes
    ----------
    sdostreamclust_ : SDOstreamclust
        Das interne SDOstreamclust-Modell.
    
    n_features_in_ : int
        Anzahl der Features (Dimensionen) im Trainingsdatensatz.
    
    Examples
    --------
    >>> from sdostreamclust_sklearn import SDOstreamclustClusterer
    >>> import numpy as np
    >>> 
    >>> # Erstelle Clusterer
    >>> clusterer = SDOstreamclustClusterer(k=10, x=5, t_fading=10.0)
    >>> 
    >>> # Initialisiere mit Daten (optional)
    >>> X_init = np.array([[1, 2], [2, 3], [10, 11]], dtype=np.float64)
    >>> clusterer.fit(X_init)
    >>> 
    >>> # Streaming: Verarbeite einzelne Punkte
    >>> for point in new_streaming_data:
    >>>     label = clusterer.predict(point.reshape(1, -1))
    >>>     clusterer.partial_fit(point.reshape(1, -1))
    """
    
    def __init__(
        self,
        k=10,
        x=5,
        t_fading=10.0,
        chi=4,
        zeta=0.5,
        min_cluster_size=2,
        distance="euclidean",
        minkowski_p=None,
        use_brute_force=False,
    ):
        """
        Initialisiere den SDOstreamclust Clusterer.
        
        Parameters
        ----------
        k : int, default=10
            Anzahl der Observer (Modellgröße).
        x : int, default=5
            Anzahl der nächsten Nachbarn für Observations.
        t_fading : float, default=10.0
            T-Parameter für fading: f = exp(-T_fading^-1).
            Die Sampling-Rate wird automatisch als t_fading/k berechnet.
        chi : int, default=4
            Anzahl der nächsten Observer für lokale Thresholds.
        zeta : float, default=0.5
            Mixing-Parameter für globale/lokale Thresholds.
        min_cluster_size : int, default=2
            Minimale Clustergröße.
        distance : str, default="euclidean"
            Distanzmetrik.
        minkowski_p : float, optional
            p-Parameter für Minkowski-Distanz.
        use_brute_force : bool, default=False
            Wenn True, wird immer Brute-Force statt des Spatial Trees verwendet.
            Dies vermeidet Probleme mit Duplikaten, ist aber langsamer für große Datensätze.
        """
        self.k = k
        self.x = x
        self.t_fading = t_fading
        self.chi = chi
        self.zeta = zeta
        self.min_cluster_size = min_cluster_size
        self.distance = distance
        self.minkowski_p = minkowski_p
        self.use_brute_force = use_brute_force
        self.sdostreamclust_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y=None, time=None):
        """
        Initialisiere das SDOstreamclust-Modell mit Daten.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Initialisierungsdaten.
        y : Ignored
            Nicht verwendet, vorhanden für scikit-learn Kompatibilität.
        time : array-like of shape (n_samples,), optional
            Zeitstempel für jeden Datenpunkt. Wenn nicht angegeben, wird auto-increment verwendet.
        
        Returns
        -------
        self : object
            Gibt self zurück.
        """
        X = self._validate_input(X, fit=True)
        
        # Konvertiere time zu numpy array wenn gegeben
        time_array = None
        if time is not None:
            time_array = np.array(time, dtype=np.float64)
            if len(time_array) == 1:
                time_array = time_array.reshape(1)
            elif len(time_array) != len(X):
                raise ValueError("time muss die gleiche Länge wie X haben oder ein einzelner Wert sein")
        
        self.sdostreamclust_ = SDOstreamclust(
            k=self.k,
            x=self.x,
            t_fading=self.t_fading,
            chi=self.chi,
            zeta=self.zeta,
            min_cluster_size=self.min_cluster_size,
            distance=self.distance,
            minkowski_p=self.minkowski_p,
            data=X,
            time=time_array[0:1] if time_array is not None and len(time_array) == 1 else None,
            use_brute_force=self.use_brute_force,
        )
        
        return self
    
    def partial_fit(self, X, y=None, time=None):
        """
        Aktualisiere das Modell mit neuen Streaming-Daten.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Neue Datenpunkte (können mehrere sein, werden einzeln verarbeitet).
        y : Ignored
            Nicht verwendet, vorhanden für scikit-learn Kompatibilität.
        time : array-like of shape (n_samples,), optional
            Zeitstempel für jeden Datenpunkt. Wenn nicht angegeben, wird auto-increment verwendet.
        
        Returns
        -------
        self : object
            Gibt self zurück.
        """
        if self.sdostreamclust_ is None:
            # Wenn noch nicht initialisiert, verwende fit()
            return self.fit(X, y, time)
        
        X = self._validate_input(X, fit=False)
        
        # Verarbeite jeden Punkt einzeln (Streaming)
        for i, point in enumerate(X):
            point_2d = point.reshape(1, -1)
            if time is not None:
                time_point = np.array([time[i]], dtype=np.float64)
                self.sdostreamclust_.learn(point_2d, time=time_point)
            else:
                self.sdostreamclust_.learn(point_2d)
        
        return self
    
    def predict(self, X):
        """
        Berechne Cluster-Labels für die gegebenen Datenpunkte.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datenpunkte, für die Cluster-Labels berechnet werden sollen.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster-Labels für jeden Datenpunkt. -1 bedeutet Outlier/kein Cluster.
        """
        if self.sdostreamclust_ is None:
            raise ValueError("Modell muss zuerst mit fit() initialisiert werden.")
        
        X = self._validate_input(X, fit=False)
        
        labels = []
        for point in X:
            point_2d = point.reshape(1, -1)
            label = self.sdostreamclust_.predict(point_2d)
            labels.append(label)
        
        return np.array(labels)
    
    def fit_predict(self, X, y=None, time=None, return_outlier_scores=False):
        """
        Initialisiere das Modell und berechne Cluster-Labels für die Initialisierungsdaten.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Initialisierungsdaten.
        y : Ignored
            Nicht verwendet, vorhanden für scikit-learn Kompatibilität.
        time : array-like of shape (n_samples,), optional
            Zeitstempel für jeden Datenpunkt. Wenn nicht angegeben, wird auto-increment verwendet.
        return_outlier_scores : bool, default=False
            Wenn True, werden auch Outlier-Scores zurückgegeben.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster-Labels für jeden Datenpunkt.
        outlier_scores : ndarray of shape (n_samples,), optional
            Outlier-Scores für jeden Datenpunkt (nur wenn return_outlier_scores=True).
        """
        # Wenn fit() mit time aufgerufen wird, müssen wir die Daten einzeln verarbeiten
        if time is not None:
            time_array = np.array(time, dtype=np.float64)
            if len(time_array) != len(X):
                raise ValueError("time muss die gleiche Länge wie X haben")
            
            # Initialisiere mit allen Daten (fit benötigt mindestens k Punkte)
            if len(X) > 0:
                # Verwende ersten Zeitstempel für Initialisierung
                init_time = time_array[0:1] if len(time_array) > 0 else None
                self.fit(X, y, time=init_time)
                
                # Verarbeite alle Punkte nochmal mit ihren individuellen Zeitstempeln
                # (fit() hat sie alle mit dem ersten Zeitstempel initialisiert)
                for i in range(len(X)):
                    point_2d = X[i:i+1]
                    time_point = np.array([time_array[i]], dtype=np.float64)
                    self.partial_fit(point_2d, time=time_point)
        else:
            self.fit(X, y, time=None)
        
        if return_outlier_scores:
            labels = self.predict(X)
            scores = self.predict_outlier_scores(X)
            return labels, scores
        else:
            return self.predict(X)
    
    def predict_outlier_scores(self, X):
        """
        Berechne Outlier-Scores für die gegebenen Datenpunkte.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datenpunkte, für die Outlier-Scores berechnet werden sollen.
        
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier-Scores für jeden Datenpunkt (höhere Werte = mehr Outlier).
        """
        if self.sdostreamclust_ is None:
            raise ValueError("Modell muss zuerst mit fit() initialisiert werden.")
        
        X = self._validate_input(X, fit=False)
        
        scores = []
        for point in X:
            point_2d = point.reshape(1, -1)
            score = self.sdostreamclust_.predict_outlier_score(point_2d)
            scores.append(score)
        
        return np.array(scores)
    
    def n_observers(self):
        """
        Gibt die Anzahl der Observer zurück.
        
        Returns
        -------
        n_observers : int
            Anzahl der Observer im Modell.
        """
        if self.sdostreamclust_ is None:
            raise ValueError("Modell muss zuerst mit fit() initialisiert werden.")
        return self.sdostreamclust_.x()
    
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
            f"SDOstreamclustClusterer(k={self.k}, x={self.x}, t_fading={self.t_fading:.2f}, "
            f"chi={self.chi}, zeta={self.zeta:.2f}, min_cluster_size={self.min_cluster_size})"
        )


# Beispiel-Verwendung
if __name__ == "__main__":
    print("=" * 60)
    print("SDOstreamclust Scikit-learn-ähnliche API - Beispiel")
    print("=" * 60)
    
    # Erstelle Clusterer
    clusterer = SDOstreamclustClusterer(
        k=10, x=5, t_fading=10.0,
        chi=4, zeta=0.5, min_cluster_size=2
    )
    print(f"\nClusterer: {clusterer}")
    
    # Initialisiere mit Daten
    np.random.seed(42)
    cluster1 = np.random.randn(10, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(10, 2) * 0.5 + np.array([8.0, 8.0])
    init_data = np.vstack([cluster1, cluster2])
    
    print(f"\nInitialisiere mit {len(init_data)} Datenpunkten...")
    clusterer.fit(init_data)
    print(f"✓ Modell initialisiert mit {clusterer.n_observers()} Observern")
    
    # Streaming: Verarbeite neue Punkte
    print("\nStreaming-Verarbeitung:")
    streaming_points = [
        np.array([2.0, 2.0], dtype=np.float64),    # Cluster 1
        np.array([8.0, 8.0], dtype=np.float64),    # Cluster 2
        np.array([2.5, 2.5], dtype=np.float64),    # Cluster 1
        np.array([15.0, 15.0], dtype=np.float64),  # Outlier/Neuer Cluster
    ]
    
    for i, point in enumerate(streaming_points, 1):
        point_2d = point.reshape(1, -1)
        
        # Vorher: Label berechnen
        label_before = clusterer.predict(point_2d)[0]
        
        # Punkt verarbeiten (Modell aktualisiert sich)
        clusterer.partial_fit(point_2d)
        
        # Nachher: Label berechnen
        label_after = clusterer.predict(point_2d)[0]
        
        label_str_before = f"Cluster {label_before}" if label_before >= 0 else "Outlier"
        label_str_after = f"Cluster {label_after}" if label_after >= 0 else "Outlier"
        print(f"  Punkt {i}: [{point[0]:5.1f}, {point[1]:5.1f}] "
              f"→ Label vorher: {label_str_before}, "
              f"nachher: {label_str_after}")
    
    print(f"\nFinale Anzahl Observer: {clusterer.n_observers()}")
    print("\n✓ Beispiel erfolgreich abgeschlossen!")
