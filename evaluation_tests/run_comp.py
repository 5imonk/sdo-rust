#!/usr/bin/env python3

from sklearn.preprocessing import MinMaxScaler
from river import cluster
from river import stream
import algorithms.algorithms as alg

import cvi
from TSindex import tempsil

import numpy as np
import pandas as pd

import sys
import os
from os.path import exists
import glob
import re
import ntpath
import time

import optuna
from optuna.samplers import TPESampler

from scipy.io.arff import loadarff 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import adjusted_rand_score

import matplotlib.pyplot as plt


def plotdata(x,y,pltname):

    if x.shape[1]>1:
        feats = np.sort(np.random.choice(x.shape[1], 2, replace=False))
    else:
        feats = np.sort(np.random.choice(x.shape[1], 2, replace=True))

    x = x[:,feats]
    t = np.arange(len(y))
    y = y.astype(int)
    f0 = 'f'+str(feats[0])
    f1 = 'f'+str(feats[1])

    gs_kw = dict(width_ratios=[1, 2], height_ratios=[1, 1])
    fig, axes = plt.subplot_mosaic([['left', 'upper right'],['left', 'lower right']], gridspec_kw=gs_kw, figsize=(10, 4), layout="constrained")
    #for k, ax in axd.items():
    #    annotate_axes(ax, f'axd[{k!r}]', fontsize=14)

    #fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.get_cmap('tab20', len(np.unique(y)))
    if np.sum(y==-1)==0:
        axes['left'].scatter(x[:,0], x[:,1], s=5, c=y, alpha=0.7, cmap=cmap) # ,edgecolors='w'
        axes['upper right'].scatter(t, x[:,0], s=5, c=y, alpha=0.7, cmap=cmap)
        axes['lower right'].scatter(t, x[:,1], s=5, c=y, alpha=0.7, cmap=cmap)
    else:
        axes['left'].scatter(x[y>-1,0], x[y>-1,1], s=5, c=y[y>-1], alpha=0.7, cmap=cmap) # ,edgecolors='w'
        axes['left'].scatter(x[y==-1,0], x[y==-1,1], s=5, c='black', alpha=0.7, cmap=cmap) # edgecolors='w'
        axes['upper right'].scatter(t[y>-1], x[y>-1,0], s=5, c=y[y>-1], alpha=0.7, cmap=cmap)
        axes['upper right'].scatter(t[y==-1], x[y==-1,0], s=5, c='black', alpha=0.7, cmap=cmap)
        axes['lower right'].scatter(t[y>-1], x[y>-1,1], s=5, c=y[y>-1], alpha=0.7, cmap=cmap)
        axes['lower right'].scatter(t[y==-1], x[y==-1,1], s=5, c='black', alpha=0.7, cmap=cmap)

    axes['left'].set_ylabel(f0)
    axes['left'].set_xlabel(f1)
    axes['upper right'].set_ylabel(f0)
    axes['lower right'].set_xlabel('time')
    axes['lower right'].set_ylabel(f1)
    axes['upper right'].yaxis.set_label_position("right")
    axes['lower right'].yaxis.set_label_position("right")

    plt.savefig(pltname) 
    plt.close()
    #plt.show()

## retrieve dataset from file into x,y arrays
def load_data(filename):

    print("Dataset (%d): %s" % (idf,filename))
    dataname = filename.split("/")[-1].split(".")[0]
    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    if(df_data['class'].dtypes == 'object'):
        df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))

    y = df_data['class'].to_numpy()
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


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

inpath  = sys.argv[1]
outpath = sys.argv[2]
pltpath = sys.argv[3]
reps = int(sys.argv[4])

np.random.seed(0)

print("\nData folder:",inpath)
print("Result folder:",outpath)

os.makedirs(outpath, exist_ok=True)
os.makedirs(pltpath, exist_ok=True)

res = []
df_columns=['filename','outliers','idf','algorithm','ARI','iXB','iPS','irCIP','TS','time']
df = pd.DataFrame(columns=df_columns)
pd.set_option('display.float_format', '{:.10f}'.format)

outfile = outpath + "results.csv"
if exists(outfile) == False:
    df.to_csv(outfile, sep=',')

algorithms = [
    ('STREAMKMeans', True, alg.optimize_skm, alg.StKMeans), 
    ('SDOstreamc', True, alg.optimize_sdc, alg.SDOclst),
    ('CluStream', True, alg.optimize_cls, alg.CluStream),
    ('DenStream', True, alg.optimize_dns, alg.DenStream),
    ('DBStream', True, alg.optimize_dbs, alg.DBStream),
    ('GT', True, None, alg.GT)] 

sampler = TPESampler(seed=10) 

for idf, filename in enumerate(sorted(glob.glob(os.path.join(inpath, '*.arff')), key=os.path.getsize)):
    timestamps,data,labels,n,m,k,outliers,dataname = load_data(filename)

    print("n,m,k:", n,m,k)

    #t_memory = int(n/10) # time/memory param
    blocksize = 1 # points to be processed on-the-fly
    if ('retail' in filename or 'fert' in filename):
        observers = 100
        init_block = 20 # initialitation block (in data points)
    else:
        observers = 500
        init_block = 200 # initialitation block (in data points)
    buffer_size = 10 # (in data points) for algs that work with internal updates
    training_block = int(n/10)
    if training_block < buffer_size:
        training_block = buffer_size
    print("Blocksize:", blocksize)
    print("Training block:", training_block)

    fixed_params = {}
    fixed_params['buffer_size'] = buffer_size 
    fixed_params['training_block'] = training_block
    fixed_params['n_clusters'] = k
    fixed_params['observers'] = observers
    fixed_params['init_block'] = init_block
    fixed_params['blocksize'] = blocksize

    for alg_name, param_search, objective_func, model in algorithms:

        for r in range(reps):

            print("\n",alg_name,r)

            ## param search
            print("** parameter search...", param_search)
            best_params = None
            try:
                if param_search:
                    study = optuna.create_study(direction='maximize', sampler=sampler)
                    study.optimize(lambda trial: objective_func(trial, data[:training_block,:], labels[:training_block], fixed_params), n_trials=50, n_jobs=1)
                    best_params = study.best_params
            except:
                best_params = None
            print("Best params:", best_params)

            start_time = time.time()

            alg = model(fixed_params,best_params)
            
            p = np.zeros(len(data)) 

            if alg_name == 'SDOstreamc' or alg_name == 'tpSDOsc':
                p[:init_block], _ = alg.fit_predict(data[:init_block,:])

            elif alg_name == 'DenStream':
                for e, _ in stream.iter_array(data[:init_block,:]):
                    alg.learn_one(e)

            for i in range(0,data.shape[0],blocksize):

                print(".", end='', flush=True)
                if ((i / blocksize) % 100 == 0):
                    print("Datapoints: ", i)

                chunk = data[i:(i+blocksize),:]

                if alg_name == 'CluStream' or alg_name == 'DBStream' or alg_name == 'STREAMKMeans' or alg_name =='DenStream':
                    j = i
                    for e, _ in stream.iter_array(chunk):
                        alg.learn_one(e)
                        p[j] = alg.predict_one(e)
                        j = j + 1
                    
                elif alg_name == 'SDOstreamc' or alg_name == 'tpSDOsc':
                    if i>init_block:
                        p[i:(i+blocksize)], _ = alg.fit_predict(chunk)

                else: # alg_name == 'GT':
                    p[i:(i+blocksize)] = labels[i:(i+blocksize)]

            
            end_time = time.time()

            label_encoder = LabelEncoder()
            if len(p==-1)>0:
                p[p>-1] = label_encoder.fit_transform(p[p>-1])
            else:
                p = label_encoder.fit_transform(p)


            ARI = adjusted_rand_score(labels, p)
          
            _,coeff, TS = tempsil(timestamps,data,p,s=200,kn=5*200,c=0)

            #incremental cvi
            ixb, ips, cip = cvi.XB(), cvi.PS(), cvi.rCIP()
            ixb_crit, ips_crit, cip_crit = np.zeros(len(p)),np.zeros(len(p)),np.zeros(len(p)) 
            for ix in range(len(p)):
                ixb_crit[ix] = ixb.get_cvi(data[ix, :], p[ix])   
                ips_crit[ix] = ips.get_cvi(data[ix, :], p[ix])   
                cip_crit[ix] = cip.get_cvi(data[ix, :], p[ix])   

            iXB = np.nanmean(ixb_crit)
            iPS = np.nanmean(ips_crit)
            rCIP = np.nanmean(cip_crit)

            new_row = {'filename':filename, 'outliers': outliers, 'idf':idf, 'algorithm':alg_name, 'ARI':ARI, 'iXB':iXB, 'iPS':iPS, 'irCIP':rCIP, 'TS':TS, 'time': end_time-start_time}
            df.loc[len(df)] = new_row 
            print(df.tail(1))
            df.tail(1).to_csv(outfile, sep=',', mode='a', header=False)

        plotname = pltpath + dataname + '_' + alg_name + '.png'
        plotdata(data,p,plotname)

print("Summary file:",outfile,"\n")
#df.to_csv(outfile, sep=',', mode='a', header=False)

