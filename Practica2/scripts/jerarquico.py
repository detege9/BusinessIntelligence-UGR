# -*- coding: utf-8 -*-
"""
Autor: 
    Daniel
Contenido:
    Clustering con el algoritmo jerárquico
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

from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from math import floor
from sklearn import preprocessing
import seaborn as sns

def dendograma (X, num_clusters, usadas, caso):
    #Para sacar el dendrograma en el jerárquico, no puedo tener muchos elementos.
    #Hago un muestreo aleatorio para quedarme solo con 1000, aunque lo ideal es elegir un caso de estudio que ya dé un tamaño así
    if len(X)>1000:
        X = X.sample(1000, random_state=123456)
    
    #En clustering hay que normalizar para las métricas de distancia
    X_normal = preprocessing.normalize(X, norm='l2')
    
    
    resultsK = open("./Resultados/"+caso+"/jerarquico_aglom/resultsWard.txt","w")
    
    ward = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    
    t = time.time()
    #Se crea el cluster y se devuelve la columna correspondiente.
    cluster_predict = ward.fit_predict(X_normal)
    tiempo = time.time() - t
    
    resultsK.write("Tiempo : {:.2f} segundos, ".format(tiempo)+ "\n")
    
    num_clusters = len(set(cluster_predict))
    clusters = pd.DataFrame(cluster_predict, index=X.index, columns=['cluster'])
    
    print("Tamaño de cada cluster:")
    resultsK.write("Tamaño de cada cluster: \n")
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
        resultsK.write('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters))+"\n")
    
    X_Jerarquico = pd.concat([X,clusters],axis=1)
    min_size = 10
    X_filtrado = X_Jerarquico[X_Jerarquico.groupby('cluster').cluster.transform(len) > min_size]
    num_clusters_filtrado = len(set(X_filtrado['cluster']))
    print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(num_clusters,num_clusters_filtrado,min_size,len(X),len(X_filtrado)))
    X_filtrado = X_filtrado.drop('cluster', 1)
    
    X_filtrado = X
    X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')
    linkage_array = hierarchy.ward(X_filtrado_normal)
    plt.figure(1)
    plt.clf()
    dendro = hierarchy.dendrogram(linkage_array,orientation='left') #lo pongo en horizontal para compararlo con el generado por seaborn
    #puedo usar, por ejemplo, "p=10,truncate_mode='lastp'" para cortar el dendrograma en 10 hojas
    plt.savefig("./Resultados/"+caso+"/jerarquico_aglom/Dendograma.png")
    plt.show()
    
    X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X_filtrado.index,columns=usadas)
    sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
    plt.savefig("./Resultados/"+caso+"/jerarquico_aglom/clustermapJerarquico.png")
    plt.show()

def jerarquico(X, X_normal, usadas, num_clusters, caso):
    
    dendograma(X, num_clusters, usadas, caso)
    