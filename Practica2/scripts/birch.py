# -*- coding: utf-8 -*-
"""
Autor: 
    Daniel
Contenido:
    Clustering con el algoritmo Birch
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

from sklearn.cluster import Birch
from sklearn import metrics
from math import floor
import seaborn as sns

def birch(X, X_normal, num_clust, caso):
    
    birch = Birch(n_clusters = num_clust, threshold=0.4)
    
    t = time.time()
    #Se crea el cluster y se devuelve la columna correspondiente.
    cluster_predict = birch.fit_predict(X_normal)
    tiempo = time.time() - t
    
    #labels_unique = np.unique(mean_shift.labels_)
    #num_clust = len(labels_unique)
        
    resultsB = open("./Resultados/"+caso+"/birch/resultsBirch.txt","w")
    
    print('----- Ejecutando Birch -----\n')
    resultsB.write("Tiempo : {:.2f} segundos, ".format(tiempo)+ "\n")
    #print("Número estimado de clusters: %d \n" %num_clust)
    
    metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f} ".format(metric_CH) + "\n")
    resultsB.write("Calinski-Harabaz Index: {:.3f} ".format(metric_CH)+ "\n")
    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    if len(X) > 10000:
       muestra_silhoutte = 0.2
    else:
       muestra_silhoutte = 1.0
   
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', 
                                         sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient Birch: {:.5f}".format(metric_SC) + "\n")
    resultsB.write("Silhouette Coefficient Birch  : {:.5f}".format(metric_SC) + "\n")

    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

    print("Tamaño de cada cluster:")
    resultsB.write("Tamaño de cada cluster: \n")
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
        resultsB.write('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters))+"\n")

    resultsB.close()
    
    X_birch = pd.concat([X_normal,clusters],axis=1)
    cluster_centers = X_birch.groupby('cluster').mean()
    centers = pd.DataFrame(cluster_centers,columns=list(X))
    centers_desnormal = centers.copy()

    #se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    resultFigure = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    
    figure = resultFigure.get_figure()
    figure.savefig("./Resultados/"+caso+"/birch/heatmapBirch.png", dpi=400)

    X_birch = pd.concat([X, clusters], axis=1)
    sns.set()
    variables = list(X_birch)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_birch, vars=variables, hue="cluster", palette='Set1', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("./Resultados/"+caso+"/birch/scatterBirch.png")