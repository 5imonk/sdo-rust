
import numpy as np
from SDOstreamclust import clustering
from river import cluster
from river import stream
from sklearn.metrics.cluster import adjusted_rand_score


def optimize_dbs(trial, x, y, pam):

    clustering_threshold = trial.suggest_float("clustering_threshold", 0.1, 3)
    fading_factor = trial.suggest_float("fading_factor", 0.01, 0.5)
    intersection_factor = trial.suggest_float("intersection_factor", 0.1, 0.9)
    minimum_weight = trial.suggest_float("minimum_weight", 0.1, 0.9)

    alg = cluster.DBSTREAM(clustering_threshold=clustering_threshold,fading_factor=fading_factor, cleanup_interval=pam['buffer_size'], intersection_factor=intersection_factor, minimum_weight=minimum_weight)

    p,i = np.zeros(len(y)),0
    for e, _ in stream.iter_array(x):
        alg.learn_one(e)
        p[i] = alg.predict_one(e)
        i = i + 1     

    ari = adjusted_rand_score(p,y)

    return ari


def optimize_dns(trial, x, y, pam):

    max_micro_clusters = trial.suggest_int("max_micro_clusters", 5*pam['n_clusters'], 20*pam['n_clusters'])
    decaying_factor = trial.suggest_float("decaying_factor", 0.1, 0.9)
    beta = trial.suggest_float("beta", 0.1, 0.9)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.3)

    alg = cluster.DenStream(decaying_factor=decaying_factor, beta=beta, mu=2/0.1, epsilon=epsilon, n_samples_init=pam['init_block'], stream_speed=pam['blocksize'])

    p,i = np.zeros(len(y)),0
    for e, _ in stream.iter_array(x):
        alg.learn_one(e)
        p[i] = alg.predict_one(e)
        i = i + 1     

    ari = adjusted_rand_score(p,y)

    return ari

def optimize_cls(trial, x, y, pam):

    time_window = trial.suggest_int("time_window", pam['buffer_size'], pam['training_block'])
    max_micro_clusters = trial.suggest_int("max_micro_clusters", 3*pam['n_clusters'], 20*pam['n_clusters'])
    halflife = trial.suggest_float("halflife", 0.1, 0.9) 
    micro_cluster_r_factor = trial.suggest_float("micro_cluster_r_factor", 1.5, 4) 
    sigma = trial.suggest_float("sigma", 0.1, 5)
    mu = trial.suggest_float("mu", 0, 1)

    alg = cluster.CluStream(n_macro_clusters=pam['n_clusters'], micro_cluster_r_factor=micro_cluster_r_factor, time_window=time_window, max_micro_clusters=max_micro_clusters, time_gap=pam['buffer_size'], seed=0, halflife=halflife, sigma=sigma, mu=mu)

    p,i = np.zeros(len(y)),0
    for e, _ in stream.iter_array(x):
        alg.learn_one(e)
        p[i] = alg.predict_one(e)
        i = i + 1     

    ari = adjusted_rand_score(p,y)

    return ari

def optimize_skm(trial, x, y, pam):

    halflife = trial.suggest_float("halflife", 0.1, 0.9)
    sigma = trial.suggest_float("sigma", 0.1, 5)
    mu = trial.suggest_float("mu", 0, 1)

    alg = cluster.STREAMKMeans(chunk_size=pam['buffer_size'], n_clusters=pam['n_clusters'], halflife=halflife, sigma=sigma, mu=mu, seed=0)

    p,i = np.zeros(len(y)),0
    for e, _ in stream.iter_array(x):
        alg.learn_one(e)
        p[i] = alg.predict_one(e)
        i = i + 1     

    ari = adjusted_rand_score(p,y)

    return ari

def optimize_sdc(trial, x, y, pam):
    minT = min(100, pam['training_block'])
    T = trial.suggest_int("T", minT, pam['training_block'], step=minT)
    outlier_threshold = trial.suggest_int("outlier_threshold", 2, 7)
    alg = clustering.SDOstreamclust(k=pam['observers'], T=T, rel_outlier_score=True, outlier_handling=True, outlier_threshold=outlier_threshold, input_buffer=pam['buffer_size'], seed=0, x=5)

    p = np.zeros(len(y))
    p, _ = alg.fit_predict(x)

    ari = adjusted_rand_score(p,y)

    return ari


def StKMeans(fixed_pams, params=None):
    chunk_size = fixed_pams['buffer_size']
    n_clusters = fixed_pams['n_clusters']
    alg = cluster.STREAMKMeans(chunk_size=chunk_size, n_clusters=n_clusters, halflife=params['halflife'], sigma=params['sigma'], mu=params['mu'], seed=0)
    return alg

def SDOclst(fixed_pams, params=None):
    buffer_size = fixed_pams['buffer_size']
    observers = fixed_pams['observers']
    alg = clustering.SDOstreamclust(k=observers, T=params['T'], rel_outlier_score=True, outlier_handling=True, outlier_threshold=params['outlier_threshold'], input_buffer=buffer_size, seed=0, x=5) 
    return alg

def CluStream(fixed_pams, params=None):
    n_clusters = fixed_pams['n_clusters']
    time_gap = fixed_pams['buffer_size']
    alg = cluster.CluStream(n_macro_clusters=n_clusters, micro_cluster_r_factor=params['micro_cluster_r_factor'], time_window=params['time_window'], max_micro_clusters=params['max_micro_clusters'], time_gap=time_gap, seed=0, halflife=params['halflife'], sigma=params['sigma'], mu=params['mu'])
    return alg

def DenStream(fixed_pams, params=None):
    n_samples_init = fixed_pams['init_block']
    stream_speed = fixed_pams['blocksize']
    alg = cluster.DenStream(decaying_factor=params['decaying_factor'], beta=params['beta'], mu=2/params['beta'], epsilon=params['epsilon'], n_samples_init=n_samples_init, stream_speed=stream_speed)
    return alg

def DBStream(fixed_pams, params=None):
    cleanup_interval = fixed_pams['buffer_size']
    n_clusters = fixed_pams['n_clusters']
    alg = cluster.DBSTREAM(clustering_threshold=params['clustering_threshold'],fading_factor=params['fading_factor'], cleanup_interval=cleanup_interval, intersection_factor=params['intersection_factor'], minimum_weight=params['minimum_weight'])
    return alg

def GT(fixed_pams, params=None):
    return None
