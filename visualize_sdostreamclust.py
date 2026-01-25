#!/usr/bin/env python3

from sdo import SDOstreamclust
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from scipy.io.arff import loadarff 

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import roc_auc_score

import moviepy.editor as mpy
import os

# Directory to save the frames
frames_dir = 'frames'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# retrieve dataset from file into x,y arrays
def load_data(filename):

    dataname = filename.split("/")[-1].split(".")[0]
    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    if(df_data['class'].dtypes == 'object'):
        df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))

    y = df_data['class'].str.strip().astype(int).to_numpy()
    
    # Generate time array
    t = np.arange(len(y))

    df_data.drop(columns=['class'], inplace=True)
    x = df_data.to_numpy()
    del df_data

    clusters = len(np.unique(y)) # num clusters in the GT
    outliers = np.sum(y==-1)
    if outliers > 0:
        clusters = clusters -1
    
    [n,m] = x.shape

    # normalize dataset
    x = MinMaxScaler().fit_transform(x)

    return t,x,y,n,m,clusters,outliers,dataname

def get_observers_info(classifier):
    """Extrahiert alle aktiven Observer-Positionen und Labels"""
    try:
        # #[getter] in PyO3 removes the "get_" prefix, so get_observers() is available as "observers"
        if hasattr(classifier, 'observers'):
            observers = classifier.observers
        elif hasattr(classifier, 'get_observers'):
            observers = classifier.get_observers
        else:
            raise AttributeError("observers/get_observers method not found - module may need to be recompiled")
        
        if callable(observers):
            observers = observers()
        
        if not observers or len(observers) == 0:
            return np.array([]), np.array([])
        
        obs_points = []
        obs_labels = []
        
        for data, label in observers:
            obs_points.append(data)
            obs_labels.append(label if label is not None else -1)
        
        return np.array(obs_points), np.array(obs_labels)
    except Exception as e:
        print(f"Error getting observers: {e}")
        return np.array([]), np.array([])

def get_all_observers_info(classifier):
    """Sammelt alle Observer-Informationen (aktiv + inaktiv)"""
    observers_data = []
    
    # Versuche k zu bekommen
    try:
        k = classifier.k if hasattr(classifier, 'k') else 800
    except:
        k = 800
    
    # Iteriere über mögliche Observer-Indizes
    # Observer-Indizes könnten nicht sequenziell sein, also versuchen wir bis k*2
    max_index = k * 2
    consecutive_errors = 0
    max_consecutive_errors = 10  # Stoppe nach 10 aufeinanderfolgenden Fehlern
    
    for i in range(max_index):
        try:
            info = classifier.get_observer_info(i)
            if info and len(info) >= 7:
                consecutive_errors = 0  # Reset error counter
                data, observations, age, time, is_active, label, cluster_obs = info
                
                # Berechne Label aus cluster_observations (argmax), wenn label None ist
                computed_label = label
                if label is None and cluster_obs and len(cluster_obs) > 0:
                    # argmax: Index mit maximalem Wert
                    max_val = max(cluster_obs)
                    if max_val > 0:
                        computed_label = cluster_obs.index(max_val)
                    else:
                        computed_label = -1  # Keine Cluster-Zugehörigkeit
                elif label is None:
                    computed_label = -1
                
                observers_data.append({
                    'index': i,
                    'data': data,
                    'observations': observations,
                    'age': age,
                    'time': time,
                    'is_active': is_active,
                    'label': computed_label,
                    'cluster_observations': cluster_obs,
                    'normalized_score': observations / age if age > 0 else 0.0
                })
        except Exception:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                break  # Wahrscheinlich keine weiteren Observer
    
    return observers_data

def print_observers_csv(observers_data, block_num, time, output_file='observers_info.csv'):
    """Gibt Observer-Informationen als CSV aus"""
    import csv
    
    if not observers_data:
        return
    
    # Bestimme Dimensionen
    if observers_data:
        data_dim = len(observers_data[0]['data'])
        cluster_obs_dim = len(observers_data[0]['cluster_observations']) if observers_data[0]['cluster_observations'] else 0
    
    # Header
    header = ['block', 'time', 'index', 'is_active', 'label', 'observations', 'age', 'normalized_score']
    header.extend([f'data_{i}' for i in range(data_dim)])
    if cluster_obs_dim > 0:
        header.extend([f'cluster_obs_{i}' for i in range(cluster_obs_dim)])
    
    # CSV schreiben
    mode = 'w' if block_num == 0 else 'a'
    with open(output_file, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(header)
        
        for obs in observers_data:
            row = [
                block_num,
                time,
                obs['index'],
                obs['is_active'],
                obs['label'],
                obs['observations'],
                obs['age'],
                obs['normalized_score']
            ]
            row.extend(obs['data'])
            if cluster_obs_dim > 0:
                row.extend(obs['cluster_observations'])
            writer.writerow(row)

filename = 'evaluation_tests/data/example/concept_drift.arff'
t,x,y,n,m,clusters,outliers,dataname = load_data(filename)

# Set the initial block to be of size k
first_block_size = 50
block_size = 50  # Remaining blocks will have this size

# Controls the time window of ground truth / predictions points shown at each frame: obs_T +/- (T / f_T), 
# obs_T is time of model (observer) snapshot
f_T = 20

k = 400 # Model size
T = 500 # Time Horizon (wird zu t_fading)
# Parameter-Mapping:
# T -> t_fading
# qv -> rho (Anteil inaktiver Observer, also rho = 1 - qv)
# e -> min_cluster_size
qv = 0.2
rho = 1 - qv  # rho = 0.8 bedeutet 80% aktive Observer
e = 3
min_cluster_size = e
chi_prop = 0.25
chi_min = 1
zeta = 0.7
# outlier_threshold und outlier_handling nicht direkt verfügbar - Outlier-Detection über Labels -1
x_ = 4

# Initialisiere SDOstreamclust mit dimension statt Warmup-Daten
classifier = SDOstreamclust(
    k=k, 
    x=x_, 
    t_fading=T,  # T -> t_fading
    t_sampling=None,  # Default: t_sampling = t_fading
    chi_min=chi_min,
    chi_prop=chi_prop, 
    zeta=zeta,
    min_cluster_size=min_cluster_size,  # e -> min_cluster_size
    rho=rho,  # qv -> rho (1 - qv)
    dimension=m  # Dimension statt Warmup-Daten
)

all_predic = []
all_scores = []

all_obs_points = []
all_obs_labels = []
all_obs_t = []

# Observer-Informationen vor dem ersten Block
obs_points, obs_labels = get_observers_info(classifier)
print(obs_points, obs_labels)
all_obs_points.append(obs_points)
all_obs_labels.append(obs_labels)
all_obs_t.append(t[0])

# CSV-Ausgabe aller Observer-Informationen
all_observers = get_all_observers_info(classifier)
print(all_observers)
print_observers_csv(all_observers, block_num=0, time=t[0], output_file='observers_info.csv')

# Process the first block separately 
chunk = x[:first_block_size, :]
chunk_time = t[:first_block_size]

# Punkt-für-Punkt verarbeiten statt fit_predict
for i in range(len(chunk)):
    point = chunk[i:i+1, :]  # 2D Array für learn()
    time_val = chunk_time[i]
    time_array = np.array([time_val], dtype=np.float64)
    
    label, score = classifier.learn(point, time=time_array)
    all_predic.append(label)
    all_scores.append(score)

# Observer-Informationen nach dem ersten Block
obs_points, obs_labels = get_observers_info(classifier)
all_obs_points.append(obs_points)
all_obs_labels.append(obs_labels)
all_obs_t.append(chunk_time[-1])

# CSV-Ausgabe aller Observer-Informationen
all_observers = get_all_observers_info(classifier)
print(all_observers)
print_observers_csv(all_observers, block_num=1, time=chunk_time[-1], output_file='observers_info.csv')

# Process the remaining blocks with size block_size
for i in range(first_block_size, x.shape[0], block_size):
    chunk = x[i:i + block_size, :]
    chunk_time = t[i:i + block_size]
    
    # Punkt-für-Punkt verarbeiten
    for j in range(len(chunk)):
        point = chunk[j:j+1, :]  # 2D Array für learn()
        time_val = chunk_time[j]
        time_array = np.array([time_val], dtype=np.float64)
        
        label, score = classifier.learn(point, time=time_array)
        all_predic.append(label)
        all_scores.append(score)
    
    # Observer-Informationen nach jedem Block
    obs_points, obs_labels = get_observers_info(classifier)
    all_obs_points.append(obs_points)
    all_obs_labels.append(obs_labels)
    all_obs_t.append(chunk_time[-1])
    
    # CSV-Ausgabe aller Observer-Informationen
    all_observers = get_all_observers_info(classifier)
    print(all_observers)
    block_num = (i - first_block_size) // block_size + 1
    print_observers_csv(all_observers, block_num=block_num+1, time=chunk_time[-1], output_file='observers_info.csv')

p = np.array(all_predic) # clustering labels
s = np.array(all_scores) # outlierness scores
s = -1/(s+1) # norm. to avoid inf scores

# Thresholding top outliers based on Chebyshev's inequality (88.9%)
# th = np.mean(s)+3*np.std(s)
# p[s>th]=-1

# Evaluation metrics
print("Adjusted Rand Index (clustering):", adjusted_rand_score(y,p))
# print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y<0,s))
print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y<0,p<0))

unique_predic_labels = np.unique(p)  # Unique labels from clustering predictions
# Collect all observer labels
all_obs_labels_flat = []
for labels in all_obs_labels:
    if len(labels) > 0:
        all_obs_labels_flat.extend(labels.tolist())
unique_obs_labels = np.unique(all_obs_labels_flat) if len(all_obs_labels_flat) > 0 else np.array([])
# Combine both to get all unique labels
if len(unique_obs_labels) > 0:
    all_unique_labels = np.unique(np.concatenate([unique_predic_labels, unique_obs_labels]))
else:
    all_unique_labels = unique_predic_labels.copy()
if -1 not in all_unique_labels:
    all_unique_labels = np.append(all_unique_labels, -1)

le = LabelEncoder().fit(all_unique_labels)
p = le.transform(p) -1

num_labels = len(all_unique_labels) - 1  # Number of unique labels (minus outlier label)
cmap = plt.get_cmap('tab20', num_labels)
# Handle case where all labels are the same (min == max)
if num_labels > 0:
    norm = plt.Normalize(vmin=0, vmax=max(1, num_labels-1))
else:
    norm = plt.Normalize(vmin=0, vmax=1)

# Define marker shapes, which will cycle if the number of labels exceeds the number of available shapes
marker_shapes = ['.', 'o', 's', 'd', '^', 'v', '<', '>', 'h', 'p', '*', '+', '1', '2', '3', '4', '8', 'P', 'D', 'H']
num_shapes = len(marker_shapes)

num_gt_labels = len(np.unique(y[y>-1])) if len(y[y>-1]) > 0 else 1
cmap_gt = plt.get_cmap('Dark2', num_gt_labels)
# Handle case where all labels are the same (min == max)
if num_gt_labels > 0:
    norm_gt = plt.Normalize(vmin=0, vmax=max(1, num_gt_labels-1))
else:
    norm_gt = plt.Normalize(vmin=0, vmax=1)

frame_files = []

with_pred = False

# Plot and save each frame
for idx, (obs_points, obs_labels, obs_t) in enumerate(zip(all_obs_points, all_obs_labels, all_obs_t)):

    # Plot filtered points with corresponding shapes
    time_min = obs_t - T/f_T
    time_max = obs_t + T/f_T
    mask = (t >= time_min) & (t <= time_max)
    filtered_points = x[mask]
    filtered_labels = p[mask].astype(int)
    filtered_gt_labels = y[mask].astype(int)

    if with_pred:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for label in np.unique(filtered_labels):
            if label != -1:
                shape = marker_shapes[label % num_shapes]  # Use filtered_labels for marker shapes
                axes[0].scatter(filtered_points[filtered_labels == label, 0], 
                                filtered_points[filtered_labels == label, 1], 
                                c=filtered_labels[filtered_labels == label], 
                                cmap=cmap, 
                                norm=norm, 
                                s=5, 
                                marker=shape)
            else:
                axes[0].scatter(filtered_points[filtered_labels==-1, 0], filtered_points[filtered_labels==-1, 1], 
                        c='black', s=5, marker='x', label='Outliers')

        #  scatter_filtered = axes[0].scatter(filtered_points[filtered_labels!=-1, 0], filtered_points[filtered_labels!=-1, 1], c=filtered_labels[filtered_labels!=-1], cmap=cmap, norm=norm, s=10, marker='.')
        
        axes[0].set_title(f'Predictions at Time: {obs_t} +/- {T/f_T}')
        axes[0].set_xlabel('Feature 0')
        axes[0].set_ylabel('Feature 1')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)

        # Plot observers in the second subplot (axes[1])
        if len(obs_points) > 0 and len(obs_labels) > 0:
            points = np.array(obs_points)
            # Ensure labels are in the correct format for LabelEncoder
            obs_labels_int = obs_labels.astype(int)
            # Transform labels using the encoder (handles missing labels gracefully)
            try:
                labels = le.transform(obs_labels_int) - 1
            except ValueError:
                # If some labels are not in encoder, use them as-is (shouldn't happen, but safety check)
                labels = obs_labels_int

            for label in np.unique(labels):
                if label != -1:
                    shape = marker_shapes[label % num_shapes]  # Use filtered_labels for marker shapes
                    axes[1].scatter(points[labels == label, 0], 
                                    points[labels == label, 1], 
                                    c=labels[labels == label], 
                                    cmap=cmap, 
                                    norm=norm, 
                                    s=5, 
                                    marker=shape)


        # scatter_obs = axes[1].scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap, norm=norm, s=10, marker='.')
        axes[1].set_title(f'Observers at Time: {obs_t}')
        axes[1].set_xlabel('Feature 0')
        axes[1].set_ylabel('Feature 1')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)

        
        for label in np.unique(filtered_gt_labels):
            if label != -1:
                shape = marker_shapes[label % num_shapes]  # Use filtered_labels for marker shapes
                axes[2].scatter(filtered_points[filtered_gt_labels!=-1, 0], 
                                filtered_points[filtered_gt_labels!=-1, 1], 
                                c=filtered_gt_labels[filtered_gt_labels!=-1], 
                                cmap=cmap_gt, 
                                norm=norm_gt,
                                s=5, 
                                marker=shape)
            else:
                axes[2].scatter(filtered_points[filtered_gt_labels==-1, 0], filtered_points[filtered_gt_labels==-1, 1], 
                        c='black', s=15, marker='x', label='Outliers')
        axes[2].set_title(f'Ground Truth at Time: {obs_t} +/- {T/f_T}')
        axes[2].set_xlabel('Feature 0')
        axes[2].set_ylabel('Feature 1')
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Ground truth plot
        for label in np.unique(filtered_gt_labels):
            if label != -1:
                shape = marker_shapes[label % num_shapes]
                axes[0].scatter(filtered_points[filtered_gt_labels == label, 0], 
                                filtered_points[filtered_gt_labels == label, 1], 
                                c=filtered_gt_labels[filtered_gt_labels == label], 
                                cmap=cmap_gt, 
                                norm=norm_gt, 
                                s=5, 
                                marker=shape)
            else:
                axes[0].scatter(filtered_points[filtered_gt_labels == -1, 0], 
                                filtered_points[filtered_gt_labels == -1, 1], 
                                c='black', s=15, marker='x', label='Outliers')

        axes[0].set_title(f'Ground Truth at Time: {obs_t} +/- {T/f_T}')
        axes[0].set_xlabel('f0')
        axes[0].set_ylabel('f1')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)

        # Plot observers in the second subplot (axes[1])
        if len(obs_points) > 0 and len(obs_labels) > 0:
            points = np.array(obs_points)
            # Ensure labels are in the correct format for LabelEncoder
            obs_labels_int = obs_labels.astype(int)
            # Transform labels using the encoder (handles missing labels gracefully)
            try:
                labels = le.transform(obs_labels_int) - 1
            except ValueError:
                # If some labels are not in encoder, use them as-is (shouldn't happen, but safety check)
                labels = obs_labels_int

            for label in np.unique(labels):
                if label != -1:
                    shape = marker_shapes[label % num_shapes]
                    axes[1].scatter(points[labels == label, 0], 
                                    points[labels == label, 1], 
                                    c=labels[labels == label], 
                                    cmap=cmap, 
                                    norm=norm, 
                                    s=5, 
                                    marker=shape)

        axes[1].set_title(f'Observers at Time: {obs_t}')
        axes[1].set_xlabel('f0')
        axes[1].set_ylabel('f1')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)

    plt.tight_layout()
    # Save the frame
    frame_file = os.path.join(frames_dir, f'frame_{idx:04d}.png')
    plt.savefig(frame_file)
    frame_files.append(frame_file)

    frame_file_eps = os.path.join(frames_dir, f'frame_{idx:04d}.eps')
    plt.savefig(frame_file_eps)  # This saves the file as EPS
    plt.close(fig)

if with_pred:
    fig = plt.figure(figsize=(18,6))
    # cmap = plt.get_cmap('tab20', len(np.unique(p)))
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        # Handle case where all predictions are outliers or all same label
        if len(p[p>-1]) > 0:
            unique_pred = np.unique(p[p>-1])
            if len(unique_pred) > 1:
                ax.scatter3D(t[p>-1], x[p>-1,0], x[p>-1,1], s=5, c=p[p>-1], cmap=cmap, norm=norm)
            else:
                # All same label - use single color
                ax.scatter3D(t[p>-1], x[p>-1,0], x[p>-1,1], s=5, c='blue')
        if len(p[p==-1]) > 0:
            ax.scatter3D(t[p==-1], x[p==-1,0], x[p==-1,1], s=5, c='black')
        ax.view_init(azim=280+i*30, elev=20)
        ax.set_xlabel('time')
        ax.set_ylabel('f0')
        ax.set_zlabel('f1')

    # Plotting ground truth
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+4, projection='3d')
        # Handle case where all ground truth labels are same
        if len(y[y>-1]) > 0:
            unique_gt = np.unique(y[y>-1])
            if len(unique_gt) > 1:
                ax.scatter3D(t[y>-1], x[y>-1,0], x[y>-1,1], s=5, c=y[y>-1], cmap=cmap_gt, norm=norm_gt)
            else:
                # All same label - use single color
                ax.scatter3D(t[y>-1], x[y>-1,0], x[y>-1,1], s=5, c='green')
        if len(y[y==-1]) > 0:
            ax.scatter3D(t[y==-1], x[y==-1,0], x[y==-1,1], s=5, c='black')
        ax.view_init(azim=280+i*30, elev=20)
        ax.set_xlabel('time')
        ax.set_ylabel('f0')
        ax.set_zlabel('f1')
else:
    fig = plt.figure(figsize=(12,6))
    for i in range(2):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        # Handle case where all predictions are outliers or all same label
        if len(p[p>-1]) > 0:
            unique_pred = np.unique(p[p>-1])
            if len(unique_pred) > 1:
                ax.scatter3D(t[p>-1], x[p>-1,0], x[p>-1,1], s=5, c=p[p>-1], cmap=cmap, norm=norm)
            else:
                # All same label - use single color
                ax.scatter3D(t[p>-1], x[p>-1,0], x[p>-1,1], s=5, c='blue')
        if len(p[p==-1]) > 0:
            ax.scatter3D(t[p==-1], x[p==-1,0], x[p==-1,1], s=5, c='black')
        ax.view_init(azim=280+i*45, elev=20)
        ax.set_xlabel('time')
        ax.set_ylabel('f0')
        ax.set_zlabel('f1')

    # Plotting ground truth
    for i in range(2):
        ax = fig.add_subplot(2, 2, i+3, projection='3d')
        # Handle case where all ground truth labels are same
        if len(y[y>-1]) > 0:
            unique_gt = np.unique(y[y>-1])
            if len(unique_gt) > 1:
                ax.scatter3D(t[y>-1], x[y>-1,0], x[y>-1,1], s=5, c=y[y>-1], cmap=cmap_gt, norm=norm_gt)
            else:
                # All same label - use single color
                ax.scatter3D(t[y>-1], x[y>-1,0], x[y>-1,1], s=5, c='green')
        if len(y[y==-1]) > 0:
            ax.scatter3D(t[y==-1], x[y==-1,0], x[y==-1,1], s=5, c='black')
        ax.view_init(azim=280+i*45, elev=20)
        ax.set_xlabel('time')
        ax.set_ylabel('f0')
        ax.set_zlabel('f1')

plt.tight_layout()

# Save the frame
frame_file = os.path.join(frames_dir, f'frame_{idx+1:04d}.png')
plt.savefig(frame_file)
frame_files.append(frame_file)

frame_file_eps = os.path.join(frames_dir, f'frame_{idx+1:04d}.eps')
plt.savefig(frame_file_eps)  # Save as EPS
plt.close(fig)

# Create a video from the saved frames
clip = mpy.ImageSequenceClip(frame_files, fps=10)
video_file = 'conglomerate_drift.mp4'
clip.write_videofile(video_file, codec='libx264')

# Clean up frames
for frame_file in frame_files:
    os.remove(frame_file)

print(f'Video saved as {video_file}')

fig = plt.figure(figsize=(12,6))

for i in range(2):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    # Handle case where all predictions are outliers or all same label
    if len(p[p>-1]) > 0:
        unique_pred = np.unique(p[p>-1])
        if len(unique_pred) > 1:
            ax.scatter3D(t[p>-1], x[p>-1,0], x[p>-1,1], s=5, c=p[p>-1], cmap=cmap, norm=norm)
        else:
            # All same label - use single color
            ax.scatter3D(t[p>-1], x[p>-1,0], x[p>-1,1], s=5, c='blue')
    if len(p[p==-1]) > 0:
        ax.scatter3D(t[p==-1], x[p==-1,0], x[p==-1,1], s=5, c='black')
    ax.view_init(azim=280+i*45, elev=20)
    ax.set_xlabel('time')
    ax.set_ylabel('f0')
    ax.set_zlabel('f1')

plt.tight_layout()

frame_file_eps = os.path.join(frames_dir, f'frame_pred.eps')
plt.savefig(frame_file_eps)  # Save as EPS
plt.close(fig)

fig = plt.figure(figsize=(12,6))

# Plotting ground truth
for i in range(2):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    # Handle case where all ground truth labels are same
    if len(y[y>-1]) > 0:
        unique_gt = np.unique(y[y>-1])
        if len(unique_gt) > 1:
            ax.scatter3D(t[y>-1], x[y>-1,0], x[y>-1,1], s=5, c=y[y>-1], cmap=cmap_gt, norm=norm_gt)
        else:
            # All same label - use single color
            ax.scatter3D(t[y>-1], x[y>-1,0], x[y>-1,1], s=5, c='green')
    if len(y[y==-1]) > 0:
        ax.scatter3D(t[y==-1], x[y==-1,0], x[y==-1,1], s=5, c='black')
    ax.view_init(azim=280+i*45, elev=20)
    ax.set_xlabel('time')
    ax.set_ylabel('f0')
    ax.set_zlabel('f1')

plt.tight_layout()

frame_file_eps = os.path.join(frames_dir, f'frame_gt.eps')
plt.savefig(frame_file_eps)  # Save as EPS
plt.close(fig)
