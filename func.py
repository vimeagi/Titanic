import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

# <--Очистка данных от мусорных колонок и добавление новых признаков-->#

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
        train_data['Deck'] = train_data['Deck'].replace({
    'T': 'No_info',
    'G': 'No_info'
    })
    train_data = train_data.drop(columns=['Cabin','Ticket'],errors='ignore')
    train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
    train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    return train_data

#<--Разделение данных на признаки и таргет-->#

def sech(data,target):
    X_train = data.drop(columns=target)
    y_train = data[target]
    return (X_train, y_train)
    
#<--Encoder-ы для преобразования cat данных-->#

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

#<-- Стандартизация данных с помощью scaler-а-->#

def normalise_train_test(X_train, X_test, method=1):
    if method == 1:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler
        
    ###MODELS###
    
# <--Knn-->#
def knn(train_x,test_x,train_y,test_y,mode=1,nei = 3,metrics=None):

    if mode == 1:
        model = KNeighborsClassifier(n_neighbors=nei)
    else:
        model = KNeighborsRegressor(n_neighbors=nei)
    model.fit(train_x,train_y)
    pred_test = model.predict(test_x)
    pred_train = model.predict(train_x)
    if metrics is None:
        if mode == 1:
            print(f'Accuracy_train_score: {accuracy_score(train_y,pred_train):.4f}')
            print(f'Accuracy_test_score: {accuracy_score(test_y,pred_test):.4f}')
        else:
            print(f'MSE_train_score: {mean_squared_error(train_y,pred_train):.4f}')
            print(f'MSE_test_score: {mean_squared_error(test_y,pred_test):.4f}')
    else:
        for metric in metrics:
            print(f'{metric.__name__}_train:  {metric(train_y,pred_train):.4f}')
            print(f'{metric.__name__}_test:  {metric(test_y,pred_test):.4f}')
    return model

# <--LogisticRegression-->#
def logReg(train_x,test_x,train_y,test_y,C=1.0,l1_ratio=None,max_iter=1000,metrics=None):
    if l1_ratio is None:
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            solver='lbfgs'
        )
    else:
        model = LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            random_state=42,
            solver='saga'
        )
    model.fit(train_x,train_y)
    pred_test = model.predict(test_x)
    pred_train = model.predict(train_x)
    if metrics is None:
        print(f'Accuracy_train_score: {accuracy_score(train_y,pred_train):.4f}')
        print(f'Accuracy_test_score: {accuracy_score(test_y,pred_test):.4f}')
    else:
        for metric in metrics:
            print(f'{metric.__name__}_train:  {metric(train_y,pred_train):.4f}')
            print(f'{metric.__name__}_test:  {metric(test_y,pred_test):.4f}')
    return model


def default_tree(train_x,test_x,train_y,test_y,mode=1,max_depth=5,min_samples_split=20,min_samples_leaf=10,metrics=None):
    if mode == 1:
        model = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,random_state=42)
    else:
        model = DecisionTreeRegressor(max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,random_state=42)
    model.fit(train_x,train_y)
    pred_test = model.predict(test_x)
    pred_train = model.predict(train_x)
    if metrics is None:
        if mode == 1:
            print(f'Accuracy_train_score: {accuracy_score(train_y,pred_train):.4f}')
            print(f'Accuracy_test_score: {accuracy_score(test_y,pred_test):.4f}')
        else:
            print(f'MSE_train_score: {mean_squared_error(train_y,pred_train):.4f}')
            print(f'MSE_test_score: {mean_squared_error(test_y,pred_test):.4f}')
    else:
        for metric in metrics:
            print(f'{metric.__name__}_train:  {metric(train_y,pred_train):.4f}')
            print(f'{metric.__name__}_test:  {metric(test_y,pred_test):.4f}')
    return model