import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import  StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE="model.pkl"
PIPLINE_FILE="pipeline.pkl"

def build_pipeline(num_attrib,cat_attrib):
    num_pipline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())
    ])

    cat_pipline=Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    #constructing full pipeline
    full_pipeline=ColumnTransformer([
        ("num",num_pipline,num_attrib),
        ("cat",cat_pipline,cat_attrib)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    #train the model
    housing= pd.read_csv("housing.csv",encoding="utf-8")
    #startisfied test set
    housing["income_cat"]=pd.cut(housing["median_income"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index,test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv",index=False) 
        housing=housing.loc[train_index].drop("income_cat",axis=1)
        
    housing_labels=housing["median_house_value"].copy()
    housing_features=housing.drop("median_house_value",axis=1)
    #seperate numerical and categorical columns
    num_attrib=housing_features.drop("ocean_proximity",axis=1).columns.to_list()
    cat_attrib=["ocean_proximity"]
    
    pipeline=build_pipeline(num_attrib,cat_attrib)
    housing_prepared=pipeline.fit_transform(housing_features)
    
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPLINE_FILE)
    print("model is train")
else:
    #lets do infrence
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPLINE_FILE)
    input_data=pd.read_csv('input.csv')
    transformed_input=pipeline.transform(input_data)
    predictions=model.predict(transformed_input)
    input_data['median_house_value']=predictions
    input_data.to_csv("output.csv")
    print("infrence is complete,results saved to output.csv") 
