# -*- coding: utf-8 -*-
"""
Autor: 
    Daniel
Contenido:
    Clustering con el algoritmo KMeans
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

from sklearn.cluster import KMeans
from sklearn import metrics
from math import floor
import seaborn as sns
sns.set()

def Elbow(X_normal,caso):
    predicts = []
    for i in range (1, 11):
        kmeans = KMeans (n_clusters=i, init='k-means++', random_state = 123456)
        kmeans.fit(X_normal)
        predicts.append(kmeans.inertia_)
    plt.plot(range(1,11),predicts)
    plt.title('Elbow')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inertia')
    plt.savefig("./Resultados/"+caso+"/kmeans/ElbowMethod.png")
    plt.show()
    
    
def kMeans(X, X_normal, caso):
    
    Elbow(X_normal,caso)
    
    num_clust = input ("¿Cuál quieres que sea el núm. de clusters? ")
    num_clust = int(num_clust)
    k_means = KMeans(init='k-means++', n_clusters=num_clust, n_init=5, max_iter=500)
    
    t = time.time()
    #Se crea el cluster y se devuelve la columna correspondiente.
    cluster_predict = k_means.fit_predict(X_normal)
    tiempo = time.time() - t
    
    resultsK = open("./Resultados/"+caso+"/kmeans/resultsKMeans.txt","w")
    
    print('----- Ejecutando KMeans -----\n')
    resultsK.write("Tiempo : {:.2f} segundos, ".format(tiempo)+ "\n")
    
    metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f} ".format(metric_CH) + "\n")
    resultsK.write("Calinski-Harabaz Index: {:.3f} ".format(metric_CH)+ "\n")
    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    if len(X) > 10000:
       muestra_silhoutte = 0.2
    else:
       muestra_silhoutte = 1.0
   
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', 
                                         sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient KMeans: {:.5f}".format(metric_SC) + "\n")
    resultsK.write("Silhouette Coefficient KMeans  : {:.5f}".format(metric_SC) + "\n")

    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

    print("Tamaño de cada cluster:")
    resultsK.write("Tamaño de cada cluster: \n")
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
        resultsK.write('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters))+"\n")

    resultsK.close()
    
    centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
    centers_desnormal = centers.copy()

    #se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    resultFigure = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    
    figure = resultFigure.get_figure()
    figure.savefig("./Resultados/"+caso+"/kmeans/heatmapKMeans.png", dpi=400)

    X_kmeans = pd.concat([X, clusters], axis=1)
    
    variables = list(X_kmeans)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Set1', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("./Resultados/"+caso+"/kmeans/scatterKmeans2.png")