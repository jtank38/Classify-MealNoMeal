import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *
from sklearn.svm import SVC
import tsfresh.feature_extraction.feature_calculators as tcal
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.model_selection import KFold
import sys


class Models():
    def __init__(self):

        self.model = None

    def fit(self, modelName, X_train, Y_train):

        if modelName == 'SVM':
            self.model = SVC(kernel='rbf', gamma='scale')

        elif modelName == 'AdaBoost':
            self.model = AdaBoostClassifier(n_estimators=100, random_state=0)

        elif modelName == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=220)  # 944 auto 140

        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        prediction = self.model.predict(X_test)

        return [i for i in prediction]


class getFeatures():
    def __init__(self, dataCleaned):
        self.data = dataCleaned
        self.dataTransposed = dataCleaned.T

    def fft(self):
        # get FFT coefficient values, return matrix
        f1 = [abs(list(tcal.fft_coefficient(self.data[i, :], [
            {"coeff": 1, "attr": "real"}]))[0][1]) for i in range(len(self.data))]
        f2 = [abs(list(tcal.fft_coefficient(self.data[i, :], [
            {"coeff": 2, "attr": "real"}]))[0][1]) for i in range(len(self.data))]
        f3 = [abs(list(tcal.fft_coefficient(self.data[i, :], [
            {"coeff": 3, "attr": "real"}]))[0][1]) for i in range(len(self.data))]
        f4 = [abs(list(tcal.fft_coefficient(self.data[i, :], [
            {"coeff": 4, "attr": "real"}]))[0][1]) for i in range(len(self.data))]
        f5 = [abs(list(tcal.fft_coefficient(self.data[i, :], [
            {"coeff": 5, "attr": "real"}]))[0][1]) for i in range(len(self.data))]
        return np.array([f1, f2, f3, f4, f5]).T

    def entropy(self):
        return np.array([tcal.sample_entropy(self.data[i, :]) for i in range(len(self.data))])

    def movingStd(self):
        df = pd.DataFrame(data=np.int_(self.dataTransposed[0:, 0:]))

        df1 = df.rolling(window=7).std().T

        return df1.dropna(axis=1, how='all')

    def skewness(self):
        return np.array([tcal.skewness(self.data[i, :]) for i in range(len(self.data))])

    def kurtosis(self):
        return np.array([tcal.kurtosis(self.data[i, :]) for i in range(len(self.data))])


class getPCA():
    def __init__(self, featureMat):
        self.mat = featureMat
        self.normData = StandardScaler().fit_transform(self.mat)
        self.pca_timeSeries = PCA(n_components=5)
        self.PCs = self.pca_timeSeries.fit_transform(self.normData)
        self.df = pd.DataFrame(self.PCs, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

    def pca(self):
        return self.df.values

    def results(self):
        return self.pca_timeSeries.explained_variance_ratio_



def main():

    #Read given CSV

    data = pd.read_csv('Test_updated.csv',header=None)

    datasubs = data[[i for i in range(30)]]
    getF = getFeatures(datasubs.values)

    F1 = getF.fft()
    F2 = getF.entropy()
    F4 = getF.movingStd()
    F5 = getF.kurtosis()

    X_test = getPCA(np.concatenate((F1, F2[:, None], F4, F5[:, None]), axis=1)).pca()

    svm_from_joblib = joblib.load('trainedModel.pkl')
    prediction = svm_from_joblib.predict(X_test)

    alist = data[30].values.tolist()

    for indx,i in enumerate(np.array(prediction)):
        print(i,alist[indx])


    # svm_from_joblib = joblib.load('trainedModel.pkl')
    # prediction = svm_from_joblib.predict(X_test,Y_test)
    # accuarcy = accuracy_score(Y_test, prediction, normalize=True)
    # print(accuarcy)



if __name__ == '__main__':
    main()
