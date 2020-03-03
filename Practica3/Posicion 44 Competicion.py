# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Algoritmos
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import smote_variants as sv

import seaborn as sns
import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()

'''
lectura de datos
'''
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('nepal_earthquake_tra.csv')
data_y = pd.read_csv('nepal_earthquake_labels.csv') 
data_x_tst = pd.read_csv('nepal_earthquake_tst.csv')


#se quitan las columnas que no se usan
data_x = data_x[data_x.columns[1:]]
data_x_tst = data_x_tst[data_x_tst.columns[1:]]
data_y.drop(labels=['building_id'], axis=1,inplace = True)

data_x['damage_grade'] = data_y['damage_grade']
'''
#Visualización de la cardinalidad de cada grado de destrucción.
sns.countplot(data_y['damage_grade'])
plt.title('Number of buildings with each damage grade')
plt.show()

#Visualización de la edad relacionado con el grado de destrucción.
plt.figure(figsize=(15,8))
sns.countplot(x=data_x["age"],hue=data_x["damage_grade"],palette="viridis")
plt.ylabel("no. of Bulidings")
plt.title("Age of Buildings")
plt.legend(["Low damage","Avg damage","High damage"],loc="upper right")
plt.xticks(rotation=45)
plt.show()

#Correlación
plt.figure(figsize=(10,10))
cor=data_x.corr()["damage_grade"]
cor=pd.DataFrame(cor)
sns.heatmap(cor,annot=True,cmap="viridis")
'''
'''
fs = FeatureSelector(data = data_x, labels = data_y)
fs.identify_zero_importance(task = 'regression', 
                            eval_metric = 'auc', 
                            n_iterations = 10, 
                             early_stopping = True)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']
# plot the feature importances
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
'''

'''
# checking the types of varibles in the dataset(int,float,object)
dtypes=pd.DataFrame(data_x.dtypes,columns=["Data Type"])
dtypes["Unique Values"]=data_x.nunique()
dtypes["Null Values"]=data_x.isnull().sum()
dtypes["% null Values"]=data_x.isnull().sum()/len(data_x)
dtypes.style.background_gradient(cmap='Set2',axis=0)
plt.show();'''


'''
#Preprocesamiento de la edad.

plt.hist(data_x['age'], bins=120)
age = data_x['age'].values.tolist()
new_age = [200 if x>= 200 else x for x in age]
data_x.drop(['age'], axis=1, inplace=True)
new_age = np.array(new_age)
age = (new_age - np.mean(new_age)) / np.std(new_age)
data_x['age'] = age.T

sns.distplot(data_x['age'], bins=15, kde=True)


#Preprocesamiento de los pisos.

sns.distplot(data_x['count_floors_pre_eq'], bins = 25, kde = True)
data_x['count_floors_pre_eq'].value_counts()
floors = data_x['count_floors_pre_eq'].values.tolist()
new_floors = [5 if x >= 5 else x for x in floors]
new_floors = np.array(new_floors)
floors = (new_floors - np.mean(new_floors)) / np.std(new_floors)
data_x.drop(['count_floors_pre_eq'], axis = 1, inplace = True)
data_x['count_floors_pre_eq'] = floors.T


#Preprocesamiento del area.

plt.hist(data_x['area_percentage'], bins=10)
ap = data_x['area_percentage'].values
ap = (ap - min(ap)) / (max(ap) - min(ap))
data_x.drop(['area_percentage'], axis = 1, inplace = True)
data_x['area_percentage'] = ap.T
sns.distplot(data_x['area_percentage'], bins = 10)


#Preprocesamiento

plt.hist(data_x['height_percentage'], bins = 25)
hp = data_x['height_percentage'].values
hp = (hp - min(hp)) / (max(hp) - min(hp))
data_x.drop(['height_percentage'], axis = 1, inplace = True)
data_x['height_percentage'] = hp.T
sns.distplot(data_x['height_percentage'], bins = 10)


#Familia

data_x['count_families'].value_counts()
cf = data_x['count_families'].values.tolist()
cf_new = [4 if x >= 4 else x for x in cf]
data_x.drop(['count_families'], axis = 1, inplace = True)
data_x['count_families'] = np.array(cf_new).T
'''

'''
Se convierten las variables categóricas a variables numéricas (ordinales)
'''

columnas_categoricas=['land_surface_condition', 'foundation_type', 'roof_type',
                        'ground_floor_type', 'other_floor_type', 'position',
                        'plan_configuration', 'legal_ownership_status']

# Por cada variable categóricas, se codifica haciendo uso de '.cat.codes'.
for i in columnas_categoricas:
    data_x[i]=data_x[i].astype("category")
    data_x[i]=data_x[i].cat.codes
    
    data_x_tst[i]=data_x_tst[i].astype("category")
    data_x_tst[i]=data_x_tst[i].cat.codes


data_x.drop(labels=['damage_grade'], axis=1,inplace = True)
X = data_x.values
X_tst = data_x_tst.values
y = np.ravel(data_y.values)

#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
le = preprocessing.LabelEncoder()

from sklearn.metrics import f1_score

def visualizacion (y_test, modelo_pred):
    
    cm=confusion_matrix(y_test, modelo_pred)
    conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:1','Predicted:2','Predicted:3'],
                                             index=['Actual:1','Actual:2','Actual:3'])
                                                                                
    plt.figure(figsize = (8,5))
    sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
    plt.title("confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=360)
    plt.show()

    
def do_smote (X, y):
    #SMOTE
    #print('Resampled dataset shape %s' % Counter(y))
    sm = sv.polynom_fit_SMOTE()
    X_res, y_res = sm.sample(X, y)
    #print('Resampled dataset shape %s' % Counter(y_res))
    
    return (X_res, y_res)

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        #x_train, y_train = do_smote(X[train],y[train])
        modelo = modelo.fit(X[train], y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("F1 score (tst): {:.4f}, tiempo: {:6.2f} segundos".format(f1_score(y[test],y_pred,average='micro') , tiempo))
        #visualizacion (y[test], y_pred)
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all
#------------------------------------------------------------------------


cat_indexes= [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
              28, 29, 30, 31, 32, 33, 34, 35, 36, 37]

'''
print ('Catboost')
clf = CatBoostClassifier(iterations=100, learning_rate=0.3, depth=6, loss_function='MultiClass',
                         l2_leaf_reg=3, random_seed=123456)
clf, y_test_catboost = validacion_cruzada(clf, X, y, skf, cat_indexes)

'''    
'''
print("------ XGB...")
clf = xgb.XGBClassifier(n_estimators = 300)
clf, y_test_clf = validacion_cruzada(clf,X,y,skf)
'''

print("------ LightGBM...")
clf = lgb.LGBMClassifier(objective='regression_l1',n_estimators=8250, max_bin=960, num_leaves=29, learning_rate=0.025)
clf, y_test_lgbm = validacion_cruzada(clf,X,y,skf)

'''
print ("Random Forest")
clf = RandomForestClassifier(n_estimators=500, random_state=123456, max_depth=None, n_jobs=-1)
clf, y_test_rf = validacion_cruzada (clf, X, y, skf)
'''
'''
print ("SVM")
clf = SVC();
clf, y_test_svm = validacion_cruzada (clf, X, y, skf)
'''
'''
print ("KNN")
knn = KNeighborsClassifier()
knn, y_test_knn = validacion_cruzada(knn, X, y, skf)
'''
'''
print ("GradientBoosted")
clf = GradientBoostingClassifier(n_estimators = 200)
clf, y_test_gb = validacion_cruzada (clf, X, y, skf)
'''
'''
print ("Decision Tree")
clf = DecisionTreeClassifier (min_samples_split = 25, criterion='entropy', random_state=123456,
                              min_samples_leaf = 25)
clf, y_test_decisiontree = validacion_cruzada(clf, X, y, skf)
'''

clf = clf.fit(X, y)
y_pred_tra = clf.predict(X)

print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission.csv", index=False)