import sys
import os

# Add paths for sdo module
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append('/home/simon/sdo/.venv/lib/python3.12/site-packages')

from python.sdostreamclust_sklearn import SDOstreamclustClusterer
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Lade Daten (versuche zuerst ARFF-Dateien aus evaluation_tests/data/example)
data_loaded = False
data_paths = [
    'evaluation_tests/data/example/concept_drift.arff',
    'evaluation_tests/data/example/*.arff',
    'example/dataset.csv',
]

for data_path in data_paths:
    try:
        if data_path.endswith('.arff'):
            # Lade ARFF-Datei
            import glob
            arff_files = glob.glob(data_path)
            if arff_files:
                filename = arff_files[0]  # Nimm erste gefundene Datei
                print(f"Lade ARFF-Datei: {filename}")
                arffdata = loadarff(filename)
                df_data = pd.DataFrame(arffdata[0])
                
                # Konvertiere class-Spalte falls nötig
                if df_data['class'].dtypes == 'object':
                    df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))
                
                y = df_data['class'].to_numpy()
                t = np.arange(len(y), dtype=np.float64)
                df_data.drop(columns=['class'], inplace=True)
                x = df_data.to_numpy().astype(np.float64)
                
                # Normalisiere Daten
                scaler = MinMaxScaler()
                x = scaler.fit_transform(x)
                
                # Konvertiere Labels zu int (falls String/Bytes)
                if y.dtype == object:
                    # Konvertiere Bytes zu String falls nötig
                    if isinstance(y[0], bytes):
                        y = np.array([val.decode('utf-8') for val in y])
                    # Konvertiere String-Labels zu int (-1 für Outlier bleibt -1)
                    y_int = []
                    for val in y:
                        try:
                            y_int.append(int(val))
                        except ValueError:
                            # Falls Label nicht konvertierbar, verwende LabelEncoder
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                            y_int = y
                            break
                    if len(y_int) == len(y):
                        y = np.array(y_int, dtype=int)
                else:
                    y = y.astype(int)
                
                data_loaded = True
                print(f"✓ Daten geladen: {len(x)} Punkte, {x.shape[1]} Dimensionen")
                break
        else:
            # Lade CSV-Datei
            df = pd.read_csv(data_path)
            t = df['timestamp'].to_numpy()
            x = df[['f0','f1']].to_numpy()
            y = df['label'].to_numpy()
            # Normalisiere Daten
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
            data_loaded = True
            print(f"✓ Daten geladen: {len(x)} Punkte, {x.shape[1]} Dimensionen")
            break
    except (FileNotFoundError, KeyError, ValueError) as e:
        continue

if not data_loaded:
    print("Warnung: Keine Daten gefunden. Generiere Beispieldaten...")
    np.random.seed(42)
    n_points = 1000
    t = np.arange(n_points, dtype=np.float64)
    
    # Generiere zwei Cluster und einige Outlier
    cluster1 = np.random.randn(n_points // 2, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(n_points // 2, 2) * 0.5 + np.array([8.0, 8.0])
    x = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    # Normalisiere Daten
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    # Labels: 0 für Cluster 1, 1 für Cluster 2
    y = np.concatenate([np.zeros(n_points // 2), np.ones(n_points // 2)]).astype(int)

k = 200 # Model size (reduziert für Demo)
T = 400 # Time Horizon (t_fading)
T_sampling = 250 # Sampling Interval (t_sampling)
x_neighbors = 4 # Anzahl nächster Nachbarn
chi_min = 1 # Minimum chi value
chi_prop = 0.1 # Proportion of k for chi calculation
zeta = 0.6 # Mixing-Parameter
min_cluster_size = 2 # Minimale Clustergröße

classifier = SDOstreamclustClusterer(
    k=k, 
    x=x_neighbors, 
    t_fading=T,
    t_sampling=T,
    chi_min=chi_min,
    chi_prop=chi_prop,
    zeta=zeta,
    min_cluster_size=min_cluster_size,
)

all_predic = []
all_scores = []

block_size = 1 # per-point processing
init_block = max(k, min(50, len(x))) # Initialisierungsblock (mindestens k)

# Initialisiere mit ersten Datenpunkten
if init_block > 0:
    init_data = x[:init_block]
    init_time = t[:init_block] if len(t) > 0 else None
    classifier.fit(init_data, time=init_time)
    init_labels, init_scores = classifier.fit_predict(init_data, time=init_time, return_outlier_scores=True)
    all_predic.extend(init_labels)
    all_scores.extend(init_scores)

# Streaming-Verarbeitung
for i in range(init_block, x.shape[0], block_size):
    chunk = x[i:i + block_size, :]
    chunk_time = t[i:i + block_size] if len(t) > 0 else None
    
    # Vorhersage vor dem Lernen
    labels_before = classifier.predict(chunk)
    scores_before = classifier.predict_outlier_scores(chunk)
    
    # Lerne den Punkt
    if chunk_time is not None:
        classifier.partial_fit(chunk, time=chunk_time)
    else:
        classifier.partial_fit(chunk)
    
    # Nach dem Lernen (optional, für Vergleich)
    all_predic.append(labels_before[0])
    all_scores.append(scores_before[0])

p = np.array(all_predic) # clustering labels
s = np.array(all_scores) # outlierness scores
s = -1/(s+1) # norm. to avoid inf scores

# Thresholding top outliers based on Chebyshev's inequality (88.9%)
th = np.mean(s)+3*np.std(s)
p[s>th] = -1

# Evaluation metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import roc_auc_score

print("Adjusted Rand Index (clustering):", adjusted_rand_score(y[:len(p)], p))
print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y[:len(p)]<0, s))

 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Nur für Visualisierung: LabelEncoder auf nicht-Outlier anwenden
p_vis = p.copy()
if np.sum(p_vis > -1) > 0:
    p_vis[p_vis > -1] = LabelEncoder().fit_transform(p_vis[p_vis > -1])

fig = plt.figure(figsize=(15,4))
cmap = plt.get_cmap('tab20', len(np.unique(p_vis[p_vis > -1])) if np.sum(p_vis > -1) > 0 else 1)

for i in range(3):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    if np.sum(p_vis > -1) > 0:
        ax.scatter3D(t[p_vis > -1], x[p_vis > -1, 0], x[p_vis > -1, 1], s=5, c=p_vis[p_vis > -1], cmap=cmap)
    if np.sum(p_vis == -1) > 0:
        ax.scatter3D(t[p_vis == -1], x[p_vis == -1, 0], x[p_vis == -1, 1], s=5, c='black')
    ax.view_init(azim=280+i*30, elev=20)
    ax.set_xlabel('time')
    ax.set_ylabel('f0')
    ax.set_zlabel('f1')

plt.tight_layout()
plt.savefig('demo_output.png', dpi=150, bbox_inches='tight')
print("\nVisualisierung gespeichert als 'demo_output.png'")
plt.close()
