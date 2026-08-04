import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def preprocessing(train_data,alone=True,title=True,deck=True):
    train_data = train_data.copy()
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
    if alone:
        train_data['is_Alone'] = ((train_data['SibSp'] == 0) & (train_data['Parch'] == 0)).astype(int)
    if title:
        train_data['Title'] = train_data.Name.str.extract('([A-Za-z]+)\.')
    train_data = train_data.drop(columns=['Name'])
    if deck:
        train_data['Deck'] = train_data['Cabin'].str[0].fillna('No_info')
        train_data = train_data[(train_data['Deck'] != 'T') & (train_data['Deck'] != 'G')]
    train_data = train_data.drop(columns=['Cabin','Ticket'],errors='ignore')
    train_data = train_data.dropna()
    return train_data

def sech(data,target):
    X_train = data.drop(columns=target)
    y_train = data[target]
    return (X_train, y_train)
    
def encoder(data,col,method=1):
    data = data.copy()
    if method == 1:
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(data[col])
        encoded = pd.DataFrame(encoded,columns=encoder.get_feature_names_out(col),index=data.index)
        encoded = pd.concat([data,encoded],axis=1)
        encoded = encoded.drop(columns=col)
    else:
        encoded = data.copy()
        encoder = LabelEncoder()
        for column in col:
            encoded[column] = encoder.fit_transform(data[column])
    return encoded
   
def normilise(data1,data2,method=1):
    if method == 1:
        scale = StandardScaler()
        data1 = scale.fit_transform(data1)
        data2 = scale.transform(data2)
    else:
        scale = MinMaxScaler()
        data1 = scale.fit_transform(data1)
        data2 = scale.transform(data2)
    return data1,data2
        
    