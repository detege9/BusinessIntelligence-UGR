# -*- coding: utf-8 -*-
"""
Autor: 
    Daniel
Contenido:
    Clustering con el algoritmo DBSCAN
    Práctica 2
    Inteligencia de Negocio
    Ingeniería Informática
    Universidad de Granada
"""

import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from math import floor
import seaborn as sns

def dbscan(X, X_normal, epsilon, min_samp, caso):
    
    dbscan = DBSCAN(eps=epsilon, min_samples = min_samp)
    t = time.time()
    #Se crea el cluster y se devuelve la columna correspondiente.
    cluster_predict = dbscan.fit_predict(X_normal)
    labels = dbscan.labels_
    tiempo = time.time() - t
    
    # Number of clusters in labels, ignoring noise if present.
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)
    
        
    resultsD = open("./Resultados/"+caso+"/dbscan/resultsDBSCAN.txt","w")
    
    print('----- Ejecutando DBSCAN -----\n')
    resultsD.write("Tiempo : {:.2f} segundos, ".format(tiempo)+ "\n")
    #print("Número estimado de clusters: %d \n" %num_clust)
    
    metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f} ".format(metric_CH) + "\n")
    resultsD.write("Calinski-Harabaz Index: {:.3f} ".format(metric_CH)+ "\n")
    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    if len(X) > 10000:
       muestra_silhoutte = 0.2
    else:
       muestra_silhoutte = 1.0
   
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', 
                                         sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient DBSCAN: {:.5f}".format(metric_SC) + "\n")
    resultsD.write("Silhouette Coefficient DBSCAN  : {:.5f}".format(metric_SC) + "\n")

    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

    resultsD.write("Epsilon: %f \n" %epsilon)
    resultsD.write("Min_samples: %f \n" %min_samp)
    resultsD.write("Número estimado de clusters: %d \n" %num_clusters)
    resultsD.write("Número estimado de puntos con ruido: %d \n" %num_noise)
    
    print("Número estimado de clusters: %d \n" %num_clusters)
    print("Número estimado de puntos con ruido: %d \n" %num_noise)
    print("Tamaño de cada cluster:")
    resultsD.write("Tamaño de cada cluster: \n")
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
        resultsD.write('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters))+"\n")

    resultsD.close()
    
    X_DBSCAN = pd.concat([X_normal,clusters],axis=1)
    cluster_centers = X_DBSCAN.groupby('cluster').mean()
    centers = pd.DataFrame(cluster_centers,columns=list(X))
    centers_desnormal = centers.copy()

    #se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    resultFigure = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    
    figure = resultFigure.get_figure()
    figure.savefig("./Resultados/"+caso+"/dbscan/heatmapDBSCAN.png", dpi=400)

    X_DBSCAN = pd.concat([X, clusters], axis=1)
    sns.set()
    variables = list(X_DBSCAN)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_DBSCAN, vars=variables, hue="cluster", palette='Set1', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("./Resultados/"+caso+"/dbscan/scatterDBSCAN.png")