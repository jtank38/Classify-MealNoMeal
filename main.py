import pandas as pd
import getData
from getFeatures import *
import numpy as np
from models import *
from sklearn.model_selection import KFold
from kfold import *
import sys


def main():
    a = getData.GetDirs()

    f1 = a.getDirs('MealData')
    f2 = a.getDirs('NoMealData')
    meal = np.array([])
    noMeal = np.array([])
    meal = np.concatenate([a.getMealNoMealData(i, 'Meal') for i in f1])
    noMeal = np.concatenate([a.getMealNoMealData(i) for i in f2])
    df = pd.DataFrame(data=meal)
    dfnoM = pd.DataFrame(data=noMeal)

    df[30] = [1 for i in range(len(meal))]
    dfnoM[30] = [0 for i in range(len(noMeal))]
    data = pd.concat([df, dfnoM], ignore_index=True)
    datasubs = data[[i for i in range(30)]]



    getF = getFeatures(datasubs.values)

    F1 = getF.fft()
    F2 = getF.entropy()
    F4 = getF.movingStd()
    F5 = getF.kurtosis()

    pca = getPCA(np.concatenate((F1, F2[:, None], F4, F5[:,None]), axis=1)).pca()
    #dataMatrix = np.concatenate((F1, F2[:, None], F3[:, None], F4), axis=1)

    X = pca
    Y = data[30].values

    data = np.concatenate((X,Y[:,None]),axis=1) #add labels
    execKfold(data)

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    #
    # a = Models()
    # a.fit('ANN',X_train,Y_train)
    # print(a.predict(X_test,Y_test,'ANN'))
    #RandomForest(X,Y)













main()
