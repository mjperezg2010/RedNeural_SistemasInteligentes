import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data():
    data = pd.read_csv("part3_data_val.csv")
    X = data.loc[:, data.columns != 'clase']
    Ytemp = data['clase']

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    X = pd.DataFrame(scaled)
    Y=[]
    for i in Ytemp.values:
        if i == 'pizza':
            Y.append([1,0,0,0,0])
        elif i == 'hamburguesa':
            Y.append([0, 1, 0, 0, 0])
        elif i == 'arroz_frito':
            Y.append([0, 0, 1, 0, 0])
        elif i == 'ensalada':
            Y.append([0, 0, 0, 1, 0])
        elif i == 'pollo_horneado':
            Y.append([0, 0, 0, 0, 1])

    return np.array(X.values),np.transpose(Y)


def load_data2(file):
    data = pd.read_csv(file)
    X = data.loc[:, data.columns != 'clase']
    Ytemp = data['clase']



load_data2("part4_pokemon_go_train.csv")