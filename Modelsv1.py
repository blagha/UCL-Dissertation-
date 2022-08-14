#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:06:24 2022

@author: alejandro
"""

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()
    # insert here your code

# In[1]:

from pymongo import MongoClient
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import scipy.stats as stats
import pylab

import sklearn.neighbors._base

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (OneHotEncoder, PowerTransformer, StandardScaler)
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import cross_val_score

os.chdir('/home/alejandro/Documents/Dissertation')

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# In[1]:

client = MongoClient('mongodb://admin:TErh03sdO6FbDGGDu6BpeGfPjaFmFxe7YIam0Dl5ofBQNffMPW4JOKv9I5cHcLNv@172.16.0.36:27017')
db = client.inmuebles
analitika_estandarizados = db['analitika_estandarizados']
df = pd.DataFrame(list(analitika_estandarizados.find()))

# In[1]:

#df = pd.read_csv('Data/inmuebles_venta.txt',sep = "|")
#df = pd.read_csv('Data/inmuebles_ventaV2.txt',sep = "|")
df_test = df.copy()   

# In[1]:
## Defining a dictionary to translate the names of the columns from Spanish to English. 
spanish_dict={'fecha_insercion':'ADDED_TO_SITE',
              'precio':"PRICE",
              'condicion_inmueble':'NEW_PROPERTY',
              'area_construida':'FLOOR_PLAN_AREA',
              'vetustez':'AGE',
              'habitaciones':'NUMBR_BEDROOMS',
              'baños':'NUMBR_BATHROOMS',
              'parqueaderos':'NUMBR_PARKING',
              'tipo_inmueble':'PROPERTY_TYPE',
              'enlace':'URL',
              'longitud':'LONGITUDE',
              'latitud':'LATITUDE',
              'tipo_operacion':'OPERATION_TYPE',
              'fuente':'LISTING_WEBSITE',
              'estrato':'SOCIOECONOMIC_BRACKET',
              'valor_administracion':'ADMIN_COST',   
              'direccion':'ADDRESS',
              'departamento':'REGION',
              'municipio':'CITY',
              'barrio_catastro':'DISTRICT',
              'locnombre':'BOROUGH',
              'topografia':'TOPOGRAPHY',
              'clase_via':'ROAD_TYPE',
              'estado_via':'ROAD_CONDITION',
              'estado_publicacion':'LISTING_STATUS',
              'estructura_promedio':'SCORE_STRUCTURE',
              'acabados_principales_promedio':'SCORE_FITTINGS',
              'bagno_promedio':'SCORE_BATHROOMS',
              'cocina_promedio':'SCORE_KITCHEN',
              'puntaje_promedio':'SCORE_AVERAGE',
              'ZONA_GEOECONOMICA':'PRICE_OF_LOT'}


df.rename(columns=spanish_dict,
          inplace=True)    

# In[1]:
    
#df = df_test    
## Dropping letting or renting listings. 
df.drop(df[df["OPERATION_TYPE"] != 'Venta'].index, inplace=True)
## Dropping listings outside of the city of Bogota. 
df = df[ (df['CITY'] == 'Bogotá') | (df['CITY'] == 'Bogotá D.C.')]

# In[1]:

def outlier_detection(df,var):
    df.loc[df[str(var)]=="nan",str(var)]=np.nan
    df[str(var)]=pd.to_numeric(df[str(var)])
    nas=df[df[str(var)].isnull()].index.values
    df=df.dropna(subset=[str(var)])
    Q1,Q3 = np.percentile(df[str(var)] , [25,75])
    IQR = Q3 - Q1
    ul = Q3+1.5*IQR
    ll = Q1-1.5*IQR
    outliers = df[(df[str(var)] > ul) | (df[str(var)] < ll)].index.values
    drop = np.concatenate([outliers, nas])
    return nas,outliers,drop

# In[1]:

def share_of_missing_values(df):
    #Checking share of missing and unique values to determine columns to drop
    share_missing_vals_num = (df.isnull().sum() / len(df))*100
    share_missing_vals_str = round(share_missing_vals_num,2).astype(str)+"%"
    share_missing_unique_vals_df = pd.DataFrame({"share_of_missing_values": share_missing_vals_str, "share_of_missing_values_num": share_missing_vals_num})
    share_missing_unique_vals_df['count_of_unique_values'] = df.apply(lambda row: row.nunique(dropna = True)).to_frame()
    share_missing_unique_vals_df = share_missing_unique_vals_df.sort_values(by = "share_of_missing_values_num", ascending = False)
    share_missing_unique_vals_df = share_missing_unique_vals_df.drop("share_of_missing_values_num", axis = 1)
    share_missing_unique_vals_df
    return (share_missing_unique_vals_df)
    
share_of_missing_values(df)

# In[1]:

#function to return plots for the feature
def normality(data,feature):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.kdeplot(data[feature])
    plt.subplot(1,2,2)
    stats.probplot(data[feature],plot=pylab)
    plt.show()

# In[1]:
    
## Choosing the columns to keep on the DataFrame.
columns_kept=['PRICE',"NEW_PROPERTY","FLOOR_PLAN_AREA","AGE",'NUMBR_BEDROOMS',"NUMBR_BATHROOMS",'NUMBR_PARKING',
              "PROPERTY_TYPE","LONGITUDE","LATITUDE","SOCIOECONOMIC_BRACKET","ADMIN_COST","BOROUGH",'ROAD_TYPE',
              'ROAD_CONDITION','SCORE_STRUCTURE','SCORE_FITTINGS','SCORE_BATHROOMS','SCORE_KITCHEN','PRICE_OF_LOT']  

## Dropping letting or renting listings. 
df1 = df.loc[:, columns_kept]    

# In[1]:
## Analysing NEW_PROPERTY column   

df1['NEW_PROPERTY'].value_counts()
df1['NEW_PROPERTY'] = df1['NEW_PROPERTY'].astype('category')
df1['NEW_PROPERTY'] = df1['NEW_PROPERTY'].cat.rename_categories({'Usado': 'No', 'Nuevo': 'Yes'})
df1['NEW_PROPERTY'] = np.where(df1['AGE']==0,'Yes','No')
pd.crosstab(df1['NEW_PROPERTY'],df1['AGE'])

# In[1]:
## Analysing PROPERTY_TYPE column   

df1['PROPERTY_TYPE'].value_counts()
df1['PROPERTY_TYPE'] = df1['PROPERTY_TYPE'].astype('category')
df1['PROPERTY_TYPE'] = df1['PROPERTY_TYPE'].cat.rename_categories({'Apartamento': 'Flat', 'Casa': 'House'})

# In[1]:
## Analysing SOCIOECONOMIC_BRACKET column   

df1["SOCIOECONOMIC_BRACKET"] = df1["SOCIOECONOMIC_BRACKET"].astype(str)
df1.drop(df1[df1["SOCIOECONOMIC_BRACKET"] == 'Campestre'].index, inplace=True)
df1.drop(df1[df1["SOCIOECONOMIC_BRACKET"] == '99.0'].index, inplace=True)
df1.drop(df1[df1["SOCIOECONOMIC_BRACKET"] == '99'].index, inplace=True)
df1.drop(df1[df1["SOCIOECONOMIC_BRACKET"] == '4.5'].index, inplace=True)


df1.loc[df1["SOCIOECONOMIC_BRACKET"]=="nan",'SOCIOECONOMIC_BRACKET']=np.nan
s = df1["SOCIOECONOMIC_BRACKET"].value_counts(normalize=True)
missing = df1["SOCIOECONOMIC_BRACKET"].isnull()

df1.loc[missing,'SOCIOECONOMIC_BRACKET'] = np.random.choice(s.index, size=len(df1[missing]),p=s.values)
df1['SOCIOECONOMIC_BRACKET'] = pd.to_numeric(df1['SOCIOECONOMIC_BRACKET'])
df1=df1.loc[df1['SOCIOECONOMIC_BRACKET']>0]
df1['SOCIOECONOMIC_BRACKET'].value_counts()

# In[1]:
## Analysing FLOOR_PLAN_AREA column   

df1=df1.loc[df1['FLOOR_PLAN_AREA']>0]
nas,outliers,drop=outlier_detection(df1,'FLOOR_PLAN_AREA') # Dropping outliers and NA values. 
df1.drop(drop, axis=0, inplace=True)

## Distribution Plots:
### 1) FLOOR_PLAN_AREA by PROPERTY_TYPE
sns.displot(data=df1, x="FLOOR_PLAN_AREA", kind="kde",hue="PROPERTY_TYPE")

### 2) FLOOR_PLAN_AREA by SOCIOECONOMIC_BRACKET
sns.displot(data=df1, x="FLOOR_PLAN_AREA", kind="kde",hue="SOCIOECONOMIC_BRACKET", palette="Set2",fill=True)

### 3) FLOOR_PLAN_AREA by SOCIOECONOMIC_BRACKET
sns.displot(data=df1, x="FLOOR_PLAN_AREA", kind="kde",hue="NEW_PROPERTY", palette="Set2")

### 4) Boxplot - FLOOR_PLAN_AREA by SOCIOECONOMIC_BRACKET
sns.boxplot(x="SOCIOECONOMIC_BRACKET", y="FLOOR_PLAN_AREA", data=df1)


# In[1]:
## Analysing PRICE column   

df1['PRICE']=pd.to_numeric(df1['PRICE'])
df1=df1.loc[df1['PRICE']>1000]
nas,outliers,drop=outlier_detection(df1,'PRICE')
df1.drop(drop, axis=0, inplace=True)
df1['PRICE_LOG'] = np.log1p(df1["PRICE"]) # Defining the logarithm of price 

## Distribution Plots:
### 1) PRICE by PROPERTY_TYPE
sns.displot(data=df1, x="PRICE", kind="kde",hue="PROPERTY_TYPE")

### 2) Displot - PRICE by SOCIOECONOMIC_BRACKET
sns.displot(data=df1, x="PRICE", kind="kde",hue="SOCIOECONOMIC_BRACKET", palette="Set2",fill=True)

### 2) Boxplot - PRICE by SOCIOECONOMIC_BRACKET ()
sns.boxplot(x="SOCIOECONOMIC_BRACKET", y="PRICE", data=df1)

### 3) PRICE by SOCIOECONOMIC_BRACKET
sns.displot(data=df1, x="PRICE", kind="kde",hue="NEW_PROPERTY", palette="Set2")

# In[1]:
## Creating PRICE_PER_SQM column defined as PRICE/FLOOR_PLAN_AREA    

df1["PRICE_PER_SQM"]=df1["PRICE"]/df1["FLOOR_PLAN_AREA"]
df1["LOG_PRICE_PER_SQM"]=np.log1p(df1["PRICE_PER_SQM"])
nas,outliers,drop=outlier_detection(df1,'LOG_PRICE_PER_SQM')
df1.drop(drop, axis=0, inplace=True)

sns.displot(data=df1, x="LOG_PRICE_PER_SQM", kind="kde",hue="PROPERTY_TYPE")
# In[1]:
## Analysing AGE column  

df1["AGE"]=np.where(df1["AGE"]>1000,2021-df1["AGE"],df1["AGE"])
df1["AGE"]=np.where((df1["AGE"].isnull()) & (df1["NEW_PROPERTY"]=='Nuevo'),0,df1["AGE"])
nas,outliers,drop=outlier_detection(df1,'AGE')
df1.drop(drop, axis=0, inplace=True)
df1=df1.loc[df1['AGE']>=0]

### 1) Displot - AGE by PROPERTY_TYPE
sns.displot(data=df1, x="AGE", kind="kde",hue="PROPERTY_TYPE") 

# In[1]:
## Creating AGE_PER_SQM column defined as AGE/FLOOR_PLAN_AREA    

df1["AGE_PER_SQM"]=df1["AGE"]/df1["FLOOR_PLAN_AREA"]
nas,outliers,drop=outlier_detection(df1,'AGE_PER_SQM')
df1.drop(drop, axis=0, inplace=True)


### 1) Displot - AGE by PROPERTY_TYPE
sns.boxplot(x="SOCIOECONOMIC_BRACKET", y="AGE_PER_SQM", data=df1)

# In[1]:
## DEFINE AGE_PER_SQM  
## INCLUIR ZONA_GEOECONOMICA (Precio del lote por metro cuadrado - CATASTRAL) - IMPORTANTE
## DISTANCIAS
## modelo por sklearn 


# In[1]:
## Analysing NUMBR_BEDROOMS column   

df1['NUMBR_BEDROOMS']=pd.to_numeric(df1['NUMBR_BEDROOMS'])
sns.displot(data=df1, x="NUMBR_BEDROOMS", kind="kde") 

nas,outliers,drop=outlier_detection(df1,'NUMBR_BEDROOMS')
df1.drop(drop, axis=0, inplace=True)
df1=df1.loc[df1['NUMBR_BEDROOMS']>=0]

### 1) Displot - NUMBR_BEDROOMS by PROPERTY_TYPE
sns.displot(data=df1, x="NUMBR_BEDROOMS", kind="kde",hue="PROPERTY_TYPE")

### 2) Boxplot - PRICE by NUMBR_BEDROOMS
sns.boxplot(x="NUMBR_BEDROOMS", y="PRICE", hue='SOCIOECONOMIC_BRACKET',data=df1)

# In[1]:
## Analysing NUMBR_BATHROOMS column   

df1['NUMBR_BATHROOMS']=pd.to_numeric(df1['NUMBR_BATHROOMS'])
nas,outliers,drop=outlier_detection(df1,'NUMBR_BATHROOMS')
df1.drop(drop, axis=0, inplace=True)
df1=df1.loc[df1['NUMBR_BATHROOMS']>=0]
sns.displot(data=df1, x="NUMBR_BATHROOMS", kind="kde") 


### 1) Displot - NUMBR_BATHROOMS by PROPERTY_TYPE
sns.displot(data=df1, x="NUMBR_BATHROOMS", kind="kde",hue="PROPERTY_TYPE")

### 2) Boxplot - PRICE by NUMBR_BATHROOMS
sns.boxplot(x="NUMBR_BATHROOMS", y="PRICE", hue='SOCIOECONOMIC_BRACKET',data=df1)

# In[1]:
## Analysing NUMBR_PARKING column   

df1['NUMBR_PARKING']=pd.to_numeric(df1['NUMBR_PARKING'])
nas,outliers,drop=outlier_detection(df1,'NUMBR_PARKING')
df1.drop(drop, axis=0, inplace=True)
df1=df1[df1['NUMBR_PARKING']>=0]
sns.displot(data=df1, x="NUMBR_PARKING", kind="kde") 


### 1) Displot - NUMBR_PARKING by PROPERTY_TYPE
sns.displot(data=df1, x="NUMBR_PARKING", kind="kde",hue="PROPERTY_TYPE")

### 2) Boxplot - PRICE by NUMBR_PARKING
sns.boxplot(x="NUMBR_PARKING", y="PRICE", hue='SOCIOECONOMIC_BRACKET',data=df1)

# In[1]:
## Analysing DISTRICT and BOROUGH columns   

df1['BOROUGH'].value_counts()

df1 = df1.dropna(subset=['BOROUGH'])
sns.displot(data=df1, x="PRICE_LOG", kind="kde",hue="BOROUGH")

meds=df1.groupby(['BOROUGH'])['PRICE'].median()
meds.sort_values(ascending=True, inplace=True)

### 2) Boxplot - PRICE by BOROUGH
sns.boxplot(x="PRICE", y="BOROUGH", 
            order=meds.index,
            data=df1)

# In[1]:
## Analysing ROAD_TYPE column

df1['ROAD_TYPE'].value_counts()
df1.loc[df1["ROAD_TYPE"]=="nan",'ROAD_TYPE']=np.nan
s = df1["ROAD_TYPE"].value_counts(normalize=True)
missing = df1["ROAD_TYPE"].isnull()

df1.loc[missing,'ROAD_TYPE'] = np.random.choice(s.index, size=len(df1[missing]),p=s.values)

### 1) Boxplot - PRICE by ROAD_TYPE
sns.boxplot(x="ROAD_TYPE", y="PRICE",data=df1)


share_of_missing_values(df1)

# In[1]:
## Analysing ROAD_TYPE column

df1['ROAD_CONDITION'].value_counts()
df1.loc[df1["ROAD_CONDITION"]=="nan",'ROAD_CONDITION']=np.nan
s = df1["ROAD_CONDITION"].value_counts(normalize=True)
missing = df1["ROAD_CONDITION"].isnull()

df1.loc[missing,'ROAD_CONDITION'] = np.random.choice(s.index, size=len(df1[missing]),p=s.values)

### 1) Boxplot - PRICE by ROAD_TYPE
sns.boxplot(x="ROAD_CONDITION", y="PRICE",data=df1)

share_of_missing_values(df1)

# In[1]:
## Analysing ADMIN_COST column 

import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

from sklearn.preprocessing import OrdinalEncoder
df3 = df1.drop(columns=['NEW_PROPERTY', 'PROPERTY_TYPE','BOROUGH','ROAD_TYPE','ROAD_CONDITION'])
df3_comp=df1[['NEW_PROPERTY', 'PROPERTY_TYPE','BOROUGH','ROAD_TYPE','ROAD_CONDITION']]
index = df3_comp.index


# Copy the original dataset
data = df3.copy()
# Impute
imputer = MissForest()
df_imputed = imputer.fit_transform(data)
df_imputed = pd.DataFrame(data=df_imputed, columns=data.columns).set_index(index)

# Keep only the Age column
df_admin = df1[['ADMIN_COST']].copy()
df_admin['ADMIN_COST_MISS_FOREST'] = df_imputed['ADMIN_COST']

# Obtain summary statistics
df_admin.describe().T[['mean', 'std', 'min', '50%', 'max']]

df4 = pd.concat([df_imputed, df3_comp], axis=1)

# In[1]:
## Analysing SCORE_STRUCTURE column

df4['SCORE_STRUCTURE']=pd.to_numeric(df4['SCORE_STRUCTURE'])
sns.displot(data=df4, x="SCORE_STRUCTURE", kind="kde")
normality(df4,'SCORE_STRUCTURE')


# In[1]:
## Analysing SCORE_FITTINGS column

df4['SCORE_FITTINGS']=pd.to_numeric(df4['SCORE_FITTINGS'])
sns.displot(data=df1, x="SCORE_FITTINGS", kind="kde")
normality(df4,'SCORE_FITTINGS')

# In[1]:
## Analysing SCORE_BATHROOMS column

df4['SCORE_BATHROOMS']=pd.to_numeric(df4['SCORE_BATHROOMS'])
sns.displot(data=df4, x="SCORE_BATHROOMS", kind="kde")
normality(df4,'SCORE_BATHROOMS')

# In[1]:
## Analysing SCORE_KITCHEN column

df4['SCORE_KITCHEN']=pd.to_numeric(df4['SCORE_KITCHEN'])
sns.displot(data=df4, x="SCORE_KITCHEN", kind="kde")
normality(df4,'SCORE_KITCHEN')

share_of_missing_values(df4)


# In[1]:
## Analysing ADMIN_COST column 

nas,outliers,drop=outlier_detection(df4,'ADMIN_COST')
df4.drop(drop, axis=0, inplace=True)
df4=df4[df4['ADMIN_COST']>0]
sns.displot(data=df4, x="ADMIN_COST", kind="kde")

sns.displot(data=df4, x="ADMIN_COST", kind="kde",hue="PROPERTY_TYPE")

normality(df4,'ADMIN_COST')

df4['ADMIN_COST_LOG'],parameters=stats.boxcox(df4["ADMIN_COST"])
normality(df4,'ADMIN_COST_LOG')


# In[1]:
## Analysing PRICE_OF_LOT column 
#importing necessary libraries

df4=df4[df4['PRICE_OF_LOT']>=10]
sns.displot(data=df4, x="PRICE_OF_LOT", kind="kde")

sns.displot(data=df4, x="PRICE_OF_LOT", kind="kde",hue="PROPERTY_TYPE")

sns.boxplot(x="SOCIOECONOMIC_BRACKET", y="PRICE_OF_LOT",hue='PROPERTY_TYPE',data=df4)

## 3) Scatter plot - PRICE_OF_LOT vs. PRICE

# In[1]:

normality(df4,'PRICE_OF_LOT')
# In[1]:
# Log transformation of PRICE_OF_LOT
df4['PRICE_OF_LOT_LOG'] = np.log1p(df4["PRICE_OF_LOT"]) # Defining the logarithm of price 
nas,outliers,drop=outlier_detection(df4,'PRICE_OF_LOT_LOG')
df4.drop(drop, axis=0, inplace=True)
normality(df4,'PRICE_OF_LOT_LOG')


# In[1]:
    
df4["SOCIOECONOMIC_BRACKET"].value_counts()


# In[1]:
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df4, df4["SOCIOECONOMIC_BRACKET"]):
        strat_train_set= df4.iloc[train_index]
        strat_test_set = df4.iloc[test_index]

# In[1]:
data=strat_train_set.copy() #Rename a copy of the train stratified 1 for visualization purposes 
data.to_csv (r'/home/alejandro/Documents/Dissertation/data.csv', index = None, header=True) 
# In[1]:

data = pd.read_csv('data.csv')

# In[1]:
plt.rcParams.update({'font.size': 18})

# 1: Floor plan distribution plot
fig, axs =plt.subplots(1,2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 10]})
floorplan=pd.DataFrame(data["FLOOR_PLAN_AREA"].describe())
floorplan['FLOOR_PLAN_AREA']=floorplan.values.round(1)
table = axs[0].table(cellText=floorplan.values,rowLabels=floorplan.index,colWidths = [1, 1],loc='center')
table.set_fontsize(18)
table.scale(1, 2.2)
axs[0].axis('tight')
axs[0].axis('off')
axs[1]=sns.histplot(data=data, x="FLOOR_PLAN_AREA",bins=20,kde=False,color="#1E8999",stat='density')
axs[1]=sns.kdeplot(data=data, x="FLOOR_PLAN_AREA", color='#FF5A5F', ax=axs[1])
axs[1].set_xlabel("Floorplan Area (sqm)")
axs[1].figure.savefig("Floorplan.png", dpi = 300, bbox_inches='tight')

# 2: Age distribution plot
fig, axs =plt.subplots(1,2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 10]})
floorplan=pd.DataFrame(data["AGE"].describe())
floorplan['AGE']=floorplan.values.round(1)
table = axs[0].table(cellText=floorplan.values,rowLabels=floorplan.index,colWidths = [1, 1],loc='center')
table.set_fontsize(18)
table.scale(1, 2.2)
axs[0].axis('tight')
axs[0].axis('off')
axs[1]=sns.histplot(data=data, x="AGE",bins=20,kde=False,color="#1E8999",stat='density')
axs[1]=sns.kdeplot(data=data, x="AGE", color='#FF5A5F', ax=axs[1])
axs[1].set_xlabel("Age of the property")
axs[1].figure.savefig("age.png", dpi = 300, bbox_inches='tight')

# 3: Admin Fee distribution plot
fig, axs =plt.subplots(1,2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 10]})
floorplan=pd.DataFrame(data["ADMIN_COST"].describe())
floorplan['ADMIN_COST']=floorplan.values.round(1)
table = axs[0].table(cellText=floorplan.values,rowLabels=floorplan.index,colWidths = [1, 1],loc='center')
table.set_fontsize(18)
table.scale(1, 2.2)
axs[0].axis('tight')
axs[0].axis('off')
axs[1]=sns.histplot(data=data, x="ADMIN_COST",bins=20,kde=False,color="#1E8999",stat='density')
axs[1]=sns.kdeplot(data=data, x="ADMIN_COST", color='#FF5A5F', ax=axs[1])
axs[1].set_xlabel("Administration Fee (COP)")
axs[1].figure.savefig("admin.png", dpi = 300, bbox_inches='tight')

# 4: Price of Lot distribution plot
fig, axs =plt.subplots(1,2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 10]})
floorplan=pd.DataFrame(data["PRICE_OF_LOT"].describe())
floorplan['PRICE_OF_LOT']=floorplan.values.round(1)
table = axs[0].table(cellText=floorplan.values,rowLabels=floorplan.index,colWidths = [1, 1],loc='center')
table.set_fontsize(18)
table.scale(1, 2.2)
axs[0].axis('tight')
axs[0].axis('off')
axs[1]=sns.histplot(data=data, x="PRICE_OF_LOT",bins=20,kde=False,color="#1E8999",stat='density')
axs[1]=sns.kdeplot(data=data, x="PRICE_OF_LOT", color='#FF5A5F', ax=axs[1])
axs[1].set_xlabel("Price of lot (COP)")
axs[1].figure.savefig("price_lot.png", dpi = 300, bbox_inches='tight')

# 5: Correlation Heat map
corr_df = data[['PRICE','FLOOR_PLAN_AREA','AGE','NUMBR_BEDROOMS','NUMBR_BATHROOMS',
              'NUMBR_PARKING','LONGITUDE','LATITUDE','SOCIOECONOMIC_BRACKET',
              'ADMIN_COST','SCORE_STRUCTURE','SCORE_FITTINGS','SCORE_BATHROOMS',
              'SCORE_KITCHEN','PRICE_OF_LOT']]
corr = corr_df.corr()
plt.figure(figsize=(15,15))
corr=sns.heatmap(corr, cmap="Blues", annot=True)
corr.figure.savefig("correlation.png", dpi = 300, bbox_inches='tight')

# 6: Floor plan plot Boxplot
plt.figure(figsize=(10,6))
data['FLOOR_PLAN_AREA_BINS'] = pd.qcut(data['FLOOR_PLAN_AREA'], q=5)
ax1=sns.boxplot(x="PRICE", y="FLOOR_PLAN_AREA_BINS", data=data, palette="Blues")
ax1.set_xlabel("Price (COP)")
ax1.set_ylabel("Floorplan Area (sqm)")
ax1.figure.savefig("floor_plan_box_plot.png", dpi = 300, bbox_inches='tight')

# 7: Number of bedrooms plot Boxplot
plt.figure(figsize=(10,6))
ax1=sns.boxplot(x="NUMBR_BEDROOMS", y="PRICE", data=data, palette="Blues")
ax1.set_xticklabels(ax1.get_xticklabels())
ax1.set_xlabel("Number of bedrooms")
ax1.set_ylabel("Price (COP)")
ax1.figure.savefig("bedrooms_box_plot.png", dpi = 300, bbox_inches='tight')

# In[1]:
plt.figure(figsize = (15,8))

sns.kdeplot(x = data["PRICE_LOG"],
                hue = data['SOCIOECONOMIC_BRACKET'], 
                fill = True,
                palette="crest")



# In[1]:
X_train = strat_train_set.drop(["PRICE_LOG","PRICE",'PRICE_PER_SQM','LOG_PRICE_PER_SQM','PRICE_OF_LOT_LOG','ADMIN_COST_LOG'], axis=1)
y_train= strat_train_set["LOG_PRICE_PER_SQM"]

X_test = strat_test_set.drop(["PRICE_LOG","PRICE",'PRICE_PER_SQM','LOG_PRICE_PER_SQM','PRICE_OF_LOT_LOG','ADMIN_COST_LOG'], axis=1)
y_test= strat_test_set["LOG_PRICE_PER_SQM"]

# In[1]:
numerical_features=["AGE",'AGE_PER_SQM',"NUMBR_BEDROOMS","NUMBR_BATHROOMS","NUMBR_PARKING",'LONGITUDE','LATITUDE',
                    'SCORE_STRUCTURE','SCORE_FITTINGS','SCORE_BATHROOMS','SCORE_KITCHEN']

#categorical_features=['NEW_PROPERTY','PROPERTY_TYPE',"BOROUGH",'SOCIOECONOMIC_BRACKET','ROAD_TYPE','ROAD_CONDITION']
categorical_features=['NEW_PROPERTY','PROPERTY_TYPE',"BOROUGH",'ROAD_TYPE','ROAD_CONDITION']

ordinal_features = ["SOCIOECONOMIC_BRACKET"]
to_log=['ADMIN_COST','PRICE_OF_LOT','FLOOR_PLAN_AREA']

#Build categorical preprocessor
one_hot_pipe=  make_pipeline(OneHotEncoder(handle_unknown='ignore',drop='first'))

#Build ordinal preprocessor
ordinal_pipe=  make_pipeline(OrdinalEncoder())

# Build numeric processor
numeric_pipe = make_pipeline(StandardScaler())

# Build power processor
power_pipe = make_pipeline(PowerTransformer())

preprocess_pipeline = ColumnTransformer(transformers=[("one_hot", one_hot_pipe, categorical_features),
                                                      #("ordinal", ordinal_pipe, ordinal_features),
                                                      ("numeric", numeric_pipe, numerical_features),
                                                      ("power_transform", power_pipe, to_log),] )

# Full preprocessor combined with RandomForestClassifier for feature selection.

pipeline = Pipeline(steps=[("preprocess", preprocess_pipeline),
                           ('feature_selection', SelectFromModel((RandomForestRegressor(n_estimators=500, n_jobs=-1,random_state=123)))),])

X_transformed = pipeline.fit_transform(X_train, y_train) #Fitting the pipeline. 
features_out = pipeline.get_feature_names_out(input_features=None) #Fitting the pipeline. 
print("X_transformed "+str(" shape: ")+str(X_transformed.shape))
X_test = pipeline.transform(X_test)

# In[2]:
# Estimating the 'out-of-the-box' models on the training set. 
num_folds = 10
seed = 42
scoring1 = 'r2'
scoring2 = 'neg_mean_absolute_error'
scoring3 = 'max_error'

models = []

models.append(('Linear Regression', LinearRegression()))
models.append(('Random Forest Regression', RandomForestRegressor()))
models.append(('Multi-layer Perceptron Regression', MLPRegressor(max_iter=500)))

kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)

r2={}
mae={}
max_error={}

results={}

for i, model in enumerate(models):
    name=model[0]
    cv_scores1 = cross_val_score(model[1], X_transformed, y_train, cv=kfold, scoring=scoring1)
    cv_scores2 = cross_val_score(model[1], X_transformed, y_train, cv=kfold, scoring=scoring2)
    cv_scores3 = cross_val_score(model[1], X_transformed, y_train, cv=kfold, scoring=scoring3)
    msg1 = "%s: %f, %f " % (scoring1, cv_scores1.mean(), cv_scores1.std())
    msg2 = "%s: %f, %f " % (scoring2, cv_scores2.mean(), cv_scores2.std())
    msg3 = "%s: %f, %f " % (scoring3, cv_scores3.mean(), cv_scores3.std())
    r2[name]=cv_scores1
    mae[name]=cv_scores2
    max_error[name]=cv_scores3
    results["out_of_box"]=[r2,mae,max_error]
    print("-------------------------------------")
    print(model[0])
    print(msg1)
    print(msg2)
    print(msg3)
    print("-------------------------------------")
    
# In[]:
### 2. RANDOM FOREST REGRESSOR: Hyperparameters tuning 
n_estimators = [50,100,200] # number of trees in the random forest
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()

from sklearn.model_selection import RandomizedSearchCV

rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
               n_iter = 200, cv = 5, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_transformed, y_train)

print ('Random grid: ', random_grid, '\n')
# print the best parameters
print ('Best Parameters: ', rf_random.best_params_, ' \n')

# In[0]:
### 3. MULTI-LAYER PERCEPTRON REGRESSOR: Hyperparameters tuning 
hidden_layer_sizes = [(150,100,50), (120,80,40), (100,50,30)]
max_iter = [50, 100,300,500]
activation = ['tanh', 'relu']
solver = ['sgd', 'adam']
alpha = [0.0001, 0.05]
learning_rate = ['constant','adaptive']

random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
               'max_iter': max_iter,
               'activation': activation,
               'solver': solver,
               'alpha': alpha,
               'learning_rate': learning_rate}

mlpr = MLPRegressor()

from sklearn.model_selection import RandomizedSearchCV

mlp_random = RandomizedSearchCV(estimator = mlpr,param_distributions = random_grid,
               n_iter = 200, cv = 5, verbose=2, random_state=42, n_jobs = -1)

mlp_random.fit(X_transformed, y_train)

print ('Random grid: ', random_grid, '\n')
# print the best parameters
print ('Best Parameters: ', mlp_random.best_params_, ' \n')

# In[2]:
# Estimating the tuned models on the training set. 
num_folds = 10
seed = 42
scoring1 = 'r2'
scoring2 = 'neg_mean_absolute_error'
scoring3 = 'max_error'

models = []

models.append(('Linear Regression', LinearRegression()))
models.append(('Random Forest Regression', RandomForestRegressor(n_estimators= 200, 
                                                                 min_samples_split= 6,
                                                                 min_samples_leaf= 1, 
                                                                 max_features= 'sqrt',
                                                                 max_depth= 70, 
                                                                 bootstrap = False)))
models.append(('Multi-layer Perceptron Regression', MLPRegressor(solver='adam',
                                                                 max_iter=100,
                                                                 learning_rate='adaptive',
                                                                 hidden_layer_sizes=(150,100,50),
                                                                 alpha=0.05,
                                                                 activation='relu')))

kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)

r2={}
mae={}
max_error={}

for i, model in enumerate(models):
    name=model[0]
    cv_scores1 = cross_val_score(model[1], X_transformed, y_train, cv=kfold, scoring=scoring1)
    cv_scores2 = cross_val_score(model[1], X_transformed, y_train, cv=kfold, scoring=scoring2)
    cv_scores3 = cross_val_score(model[1], X_transformed, y_train, cv=kfold, scoring=scoring3)    
    msg1 = "%s: %f, %f " % (scoring1, cv_scores1.mean(), cv_scores1.std())
    msg2 = "%s: %f, %f " % (scoring2, cv_scores2.mean(), cv_scores2.std())
    msg3 = "%s: %f, %f " % (scoring3, cv_scores3.mean(), cv_scores3.std())
    r2[name]=cv_scores1
    mae[name]=cv_scores2
    max_error[name]=cv_scores3
    results["tuned"]=[r2,mae,max_error]
    print("-------------------------------------")
    print(model[0])
    print(msg1)
    print(msg2)
    print(msg3)
    print("-------------------------------------")
    
# In[2]:

RFR = RandomForestRegressor(n_estimators= 200, 
                                     min_samples_split= 6,
                                     min_samples_leaf= 1, 
                                     max_features= 'sqrt',
                                     max_depth= 70, 
                                     bootstrap = False)
RFR.fit(X_transformed,y_train)
y_pred = RFR.predict(X_test)
scores = cross_val_score(RFR, X_transformed, y_train, cv=10)


# In[2]:
    
y_pred_exp = np.exp(y_pred)
y_test_exp = np.exp(y_test)

plt.figure(figsize=(10,10))
plt.scatter(y_test_exp, y_pred_exp, c='crimson')
p1 = max(max(y_pred_exp), max(y_test_exp))
p2 = min(min(y_pred_exp), min(y_test_exp))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

