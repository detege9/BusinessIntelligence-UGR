# -*- coding: utf-8 -*-
"""
Autor: 
    Daniel Terol 
Contenido:
    Caso 2 de estudio
    Práctica 2
    Inteligencia de Negocio
    Ingeniería Informática
    Universidad de Granada
"""

import pandas as pd
import numpy as np
import kmeans as km
import meanshift as ms
import birch as br
import dbscan as db
import jerarquico as j
import os

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

'''
for col in censo:
   missing_count = sum(pd.isnull(censo[col]))
   if missing_count > 0:
      print(col,missing_count)
#'''

#Se pueden reemplazar los valores desconocidos por un número
#censo = censo.replace(np.NaN,0)

#O imputar, por ejemplo con la media      
for col in censo:
   censo[col].fillna(censo[col].mean(), inplace=True)
      
#seleccionar casos
subset = censo.loc[((censo['EDAD']>=35) & (censo['DESEOHIJOS']==1))]
#subset = subset.loc[(censo['COD_CCAA']=='01')]

#seleccionar variables de interés para clustering
usadas = ['EDAD','EDADIDEAL', 'TEMPRELA', 'INGRESOS', 'RELIGION']
X = subset[usadas]

#Para normalizar entre 0-1. En un problema real se puede asignar un peso
#a una variable si sabemos que es más importante al resto.
X_normal = X.apply(norm_to_zero_one)

#Caso 1.
caso = "Caso3"

# KMEANS
#km.kMeans(X,X_normal,caso)

# MEAN SHIFT
#ms.meanshift(X,X_normal,caso)

# BIRCH
#dataset, datasetnorm, num_clusters, caso
#br.birch(X,X_normal,2,caso)

# DBSCAN
#dataset, datasetnorm, epsilon, min_samples, caso
#db.dbscan(X,X_normal,0.2,10,caso)

# CLUSTERING JERÁRQUICO.
#dataset, num_clusters, caso
j.jerarquico(X,X_normal, usadas, 3,caso)