
#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#%%

"""
importation et manipulation de la base de données
"""
df = pd.read_csv("/home/users/etudiant/Téléchargements/heart.csv",header = 0) 
#statistiques descriptives et manipulation
print(type(df))
print(df.shape)
print(df.head())#affichage des 5 premieres lignes de la base
print(df.columns)#affichage des noms de colomns
print(df.dtypes)#affichage du type de chaque variable
print(df.describe(include='all'))#description des données
"""
Certains indicateurs statistiques ne sont valables que pour les variables numériques
 (ex. moyenne, min, etc. pour age,tauxmax,...), et inversemment 
 pour les non-numériques (ex. top, freq, etc. pour sexe, typedouleur, ...),
 d'où les 0.00000 danscertaines situations.
"""
#%%
"""
 Variables (age et sexe)
"""
print(df['sex']) #1 homme , 0 femme
print(df.sex) # autre ecriture
print(df[['sex','age']])#accéder à un ensemble de colonnes
print(df[['sex','age']].head()) #afficher les premiéeres valeurs
print(df['age'].tail())#affichage des dernières valeurs
print(df['age'].describe())
print(df['sex'].value_counts()) #713 hommes, 312 femmes
print(df.sort_values(by='age').head())
#boucler sur l'ensemble des colonnes pour afficher le type de chaque variable
for col in df.columns:
    print(df[col].dtype) 
#%%
#avec la librairie Numpy creation de fonctions
def operation(x):
    return(x.mean())#appel de la fonction sur l'ensemble des colonnes du DataFrame
#axis = 0 ==> chaque colonne sera transmise à la fonction operation()
#la selection select_dtypes() permet d'exclure les variables non numériques
resultat = df.select_dtypes(exclude=['object']).apply(operation,axis=0)
print(resultat)    
    
#%% 
#liste des personnes de moins de 45 ans, de sexe masculin, présentant une maladie cardiaque
print(df.loc[(df['age'] < 45) & (df['sex'] =="1") & (df['target'] =="1"),:])
print(pd.crosstab(df['sex'],df['target'])) #tables croisée

"""
226 femmes sur 312  présentant une maladie cardiaque
300 hommes sur 713   présentant une maladie cardiaque

"""
#%%
"""
Visualisation graphique des variables (histogramme)
"""  
#histogramme de l'âge
df.hist(column='age') 
df.hist(column='sex')  

#histogrammes de l'âge selon le sexe
df.hist(column='age',by='sex')
#histogrammes de l'age selon target
df.hist(column='age',by='target')
#histogrammes de sexe selon target
df.hist(column='sex',by='target')

#%% 
"""
Visualisation graphique des variables (boxplot)
""" 

df.boxplot(column='age',by='sex') 
df.boxplot(column='age',by='target') 

#%%  
"""
correlation
"""
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax) 

#%% 
"""
Analyse
"""
#le nombre de gens malades et non malades
print(df['target'].value_counts()) # 1 526, 0 499
df.hist(column='target')  
#df.target.value_counts().plot(kind = 'pie', figsize = (8, 6))

#plt.legend(["malade", "Non malade "]); 
print(pd.crosstab(df.sex, df.cp) )
 
#type de douleur(cp) selon le sexe
pd.crosstab(df.sex, df.cp).plot(kind = 'bar', color = ['coral', 'lightskyblue', 'plum', 'khaki'])
plt.title('Type de cp selon le sexe')
plt.xlabel('0 = Femme,    1 = homme'); 

 
print(pd.crosstab(df.cp, df.target))
pd.crosstab(df.cp, df.target).plot(kind = 'bar', color = ['coral', 'lightskyblue', 'plum', 'khaki'])
plt.title('Type de cp selon le patient est malade ou pas')
plt.xlabel('0 = non malade,    1 = malade');  

# Creation d'une  figure
plt.figure(figsize=(10,6))

#graphique des patients qui sont malade
plt.scatter(df.age[df.target==1], 
            df.thalach[df.target==1], 
            c="tomato")

#graphique des patients qui ne sont pas malade
plt.scatter(df.age[df.target==0], 
            df.thalach[df.target==0], 
            c="lightgreen")

# Addind info
plt.title("Heart Disease w.r.t Age et Max Heart Rate (thalach)")
plt.xlabel("Age")
plt.legend(["malade", "Non malade"])
plt.ylabel("Max Heart Rate"); 
 
# Creation d'une  figure
plt.figure(figsize=(10,6))

#graphique des patients qui sont malade
plt.scatter(df.age[df.target==1], 
            df.chol[df.target==1], 
            c="r")

#graphique des patients qui ne sont pas malade
plt.scatter(df.age[df.target==0], 
            df.chol[df.target==0], 
            c="b")

# Addind info
plt.title("Heart Disease w.r.t Age et Max chol")
plt.xlabel("Age")
plt.legend(["malade", "Non malade"])
plt.ylabel("chol"); 

#fbs
df.fbs.value_counts().plot(kind = 'pie', figsize = (8, 6))
plt.legend(['fbs<120 mg/dl', 'fbs>120 mg/dl']); 
pd.crosstab(df.sex, df.fbs) 

#%% 

""" 
Machine learning: 
        étape 1: separation de la base de données en deux sous bases: train et test
        étape 2: randomforest
                 xgboost
"""
 
x = df.iloc[:, 0:-1] #table sans la variable target
print(x.head())
y=df.iloc[:,-1]#afficher la colonne target
print(y.head()) 

# étape 1/ separation de la base de données en deux sous bases: train et test

from sklearn.model_selection import train_test_split

np.random.seed(72)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) 
print(len(x_train), len(x_test), len(y_train), len(y_test) )#verifier les dimensions
#importing RandomForestClassifier


#étape 2/ 

#randomforest : Random Forest Classfier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

rf = RandomForestClassifier(n_estimators=20, random_state=2,max_depth=5)
rf.fit(x_train,y_train)
rf_predicted = rf.predict(x_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confussion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy de Random Forest:",rf_acc_score*100,'\n')
print(classification_report(y_test,rf_predicted))

#Extreme Gradient Boost XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
xgb.fit(x_train, y_train)
xgb_predicted = xgb.predict(x_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
xgb_acc_score = accuracy_score(y_test, xgb_predicted)
print("confussion matrix")
print(xgb_conf_matrix)
print("\n")
print("Accuracy de XGboost:",xgb_acc_score*100,'\n')
print(classification_report(y_test,xgb_predicted))

#variable importantes:
imp_feature = pd.DataFrame({'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'], 'Importance': xgb.feature_importances_})
plt.figure(figsize=(10,4))
plt.title("importance des variables ")
plt.xlabel("importance ")
plt.ylabel("features")
plt.barh(imp_feature['Feature'],imp_feature['Importance'],color = 'rgbkymc')
plt.show()
#%%
"""
conclusion:
Parmi les deux modeles construient on retient selui avec le modele xgboost car
l'accuracy est de 89.75.
(exang) et la douleur thoracique (cp) sont les facteurs  principaux
pour qu'un pateint ait une  crise cardiaque.
"""



 