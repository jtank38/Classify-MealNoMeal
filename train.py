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



class GetDirs():

    def getDirs(self, root):
        fileList = []
        for subdir, dirs, files in os.walk(root):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".csv"):
                    fileList.append(root + '/' + file)

        return fileList

    def missingValues(self, dfL, dfN):
        # Interpolate to remove nan values

        df_series_list = dfL.values.tolist()
        correctedDF = self.missingValuesHelper(dfL, df_series_list)
        indexes = correctedDF[correctedDF[correctedDF.columns[0]].isnull()].index.tolist()
        if len(indexes) >= 1:  # remove nans from both series
            for i in indexes:
                correctedDF = correctedDF.drop(i)
                dfN = dfN.drop(i)

        dfN_series_list = dfN.values.tolist()
        dfDateNum = self.missingValuesHelper(dfN, dfN_series_list)

        return np.array(correctedDF), np.array(dfDateNum)

    def missingValuesHelper(self, df, dfSeries):
        interpolated_data = []
        for series in dfSeries:
            cleaned_data = pd.Series(series).interpolate(method='linear', limit_direction='forward').to_list()
            interpolated_data.append(cleaned_data)

        return pd.DataFrame(interpolated_data, columns=df.columns)

    def getMealNoMealData(self, listFileNames, types=None):

        if types == 'Meal':
            df = pd.read_csv(listFileNames, names=list(range(30)))

        else:
            df = pd.read_csv(listFileNames, names=list(range(30)))

        df_series_list = df.values.tolist()
        correctedDF = self.missingValuesHelper(df, df_series_list)
        indexes = correctedDF[correctedDF[correctedDF.columns[0]].isnull()].index.tolist()
        if len(indexes) >= 1:  # remove nans from both series
            for i in indexes:
                correctedDF = correctedDF.drop(i)

        return np.array(correctedDF)


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

    def predict(self, X_test, Y_test):
        prediction = self.model.predict(X_test)

        accuarcy = accuracy_score(Y_test, prediction, normalize=True)

        otherMetrics = classification_report(Y_test, prediction, output_dict=True)[
            '1.0']  # it's binary we can us any one

        return [accuarcy, otherMetrics['f1-score'], otherMetrics['recall'], otherMetrics['precision']]


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


def execKfold(featureMat):
    kf = KFold(10, True, 1)
    testmodel1 = []
    testmodel2 = []
    testmodel3 = []
    for train_index, test_index in kf.split(featureMat):
        train, test = featureMat[train_index], featureMat[test_index]
        X_train, Y_train, X_test, Y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
        models = kfoldTrain(X_train, Y_train)
        p1, p2, p3 = testing(models, X_test, Y_test)
        testmodel1.append(p1)
        testmodel2.append(p2)
        testmodel3.append(p3)

    testmodel1 = np.array(testmodel1)
    testmodel2 = np.array(testmodel2)
    testmodel3 = np.array(testmodel3)

    result1 = np.mean(testmodel1, axis=0)
    result2 = np.mean(testmodel2, axis=0)
    result3 = np.mean(testmodel3, axis=0)

    resultlabels = ['Accuracy', 'F1-score', 'Recall', 'Precision']
    for ind, i in enumerate(list(result1)):
        print('{0} is {1} for SVM \n'.format(resultlabels[ind], i))
    print('------------------------------------------------------')

    for ind, i in enumerate(list(result2)):
        print('{0} is {1} for AdaBoost \n'.format(resultlabels[ind], i))
    print('------------------------------------------------------')

    for ind, i in enumerate(list(result3)):
        print('{0} is {1} for RandomForest \n'.format(resultlabels[ind], i))


def kfoldTrain(X_train, Y_train):
    m1 = Models()
    m1.fit('SVM', X_train, Y_train)
    m2 = Models()
    m2.fit('AdaBoost', X_train, Y_train)
    m3 = Models()
    m3.fit('RandomForest', X_train, Y_train)

    return [m1, m2, m3]


def testing(model, X_Test, Y_Test):
    testSVM = model[0].predict(X_Test, Y_Test)
    testAda = model[1].predict(X_Test, Y_Test)
    testRF = model[2].predict(X_Test, Y_Test)

    return testSVM, testAda, testRF


def main():
    a = GetDirs()

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

    pca = getPCA(np.concatenate((F1, F2[:, None], F4, F5[:, None]), axis=1)).pca()

    X = pca
    Y1 = data[30].values

    data_new = np.concatenate((X, Y1[:, None]), axis=1)  # add labels
    execKfold(data_new)

    print('-------Using Support Vector Machine-------------------')
    Y = data_new[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    m = Models()
    m.fit('SVM', X_train, Y_train)
    # svm_from_joblib = joblib.load('trainedModel.pkl')
    # prediction = svm_from_joblib.predict(X_test,Y_test)
    # accuarcy = accuracy_score(Y_test, prediction, normalize=True)
    # print(accuarcy)
    joblib.dump(m, 'trainedModel.pkl')

    print('Sucess! \n Saved model to trainedModel.pkl')


if __name__ == '__main__':
    main()
