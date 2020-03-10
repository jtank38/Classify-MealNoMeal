import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *
from sklearn.svm import SVC
import tsfresh.feature_extraction.feature_calculators as tcal
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)
from sklearn.model_selection import KFold
import joblib


class GetDirs:

    def getDirs(self, root):
        fileList = []
        for subdir, dirs, files in os.walk(root):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".csv"):
                    fileList.append(root + '/' + file)

        return fileList

    def missingValuesHelper(self, df, dfSeries):
        interpolated_data = []
        for series in dfSeries:
            cleaned_data = pd.Series(series).interpolate(method='linear', limit_direction='forward').to_list()
            interpolated_data.append(cleaned_data)

        return pd.DataFrame(interpolated_data, columns=df.columns)

    def getMealNoMealData(self, listFileNames, types=None):

        if types == 'Meal':
            df = pd.read_csv(listFileNames, names=list(range(31)))

        else:
            df = pd.read_csv(listFileNames, names=list(range(31)))

        df_series_list = df.values.tolist()
        correctedDF = self.missingValuesHelper(df, df_series_list)
        indexes = correctedDF[correctedDF[correctedDF.columns[0]].isnull()].index.tolist()
        vals = correctedDF.mean(axis=0).tolist()
        if len(indexes) >= 1:  # remove nans from both series
            for i in indexes:
                correctedDF.loc[i] = vals

        return np.array(correctedDF)


class Models:
    def __init__(self):

        self.model = None

    def fit(self, modelName, X_train, Y_train):

        if modelName == 'SVM':
            self.model = SVC(kernel='rbf', gamma='scale')

        elif modelName == 'AdaBoost':
            self.model = AdaBoostClassifier(n_estimators=100, random_state=0)

        elif modelName == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=733, max_features='sqrt', max_depth=260)  # 944 auto 140

        self.model.fit(X_train, Y_train)

    def predict(self, X_test, Y_test):
        prediction = self.model.predict(X_test)

        accuarcy = accuracy_score(Y_test, prediction, normalize=True)

        otherMetrics = classification_report(Y_test, prediction, output_dict=True)[
            '1.0']  # it's binary we can us any one

        return [accuarcy, otherMetrics['f1-score'], otherMetrics['recall'], otherMetrics['precision']]


class getFeatures:
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


def execKfold(featureMat, scorelabels):
    testmod1 = []
    testmod2 = []
    testmod3 = []

    k_f = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_ind, test_ind in k_f.split(featureMat):
        train, test = featureMat[train_ind], featureMat[test_ind]
        X_train, Y_train, X_test, Y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
        models = kfoldTrain(X_train, Y_train)
        mdt1, mdt2, mdt3 = testing(models, X_test, Y_test)
        testmod1.append(mdt1)
        testmod2.append(mdt2)
        testmod3.append(mdt3)

    testmod1 = np.array(testmod1)
    testmod2 = np.array(testmod2)
    testmod3 = np.array(testmod3)

    result1 = np.mean(testmod1, axis=0)
    result2 = np.mean(testmod2, axis=0)
    result3 = np.mean(testmod3, axis=0)

    for ind, i in enumerate(list(result1)):
        print('{0} is {1} for SVM \n'.format(scorelabels[ind], i))
    print('------------------------------------------------------')

    for ind, i in enumerate(list(result2)):
        print('{0} is {1} for AdaBoost \n'.format(scorelabels[ind], i))
    print('------------------------------------------------------')

    for ind, i in enumerate(list(result3)):
        print('{0} is {1} for RandomForest \n'.format(scorelabels[ind], i))


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
    # Get data
    a = GetDirs()

    # MealData and NoMealData directory for the csv files

    f1 = a.getDirs('MealData')
    f2 = a.getDirs('NoMealData')
    meal = np.array([])
    noMeal = np.array([])
    meal = np.concatenate([a.getMealNoMealData(i, 'Meal') for i in f1])
    noMeal = np.concatenate([a.getMealNoMealData(i) for i in f2])
    df = pd.DataFrame(data=meal)
    dfnoM = pd.DataFrame(data=noMeal)

    # Label Data 0-> No Meal 1-> Meal
    df[31] = [1 for i in range(len(meal))]
    dfnoM[31] = [0 for i in range(len(noMeal))]
    data = pd.concat([df, dfnoM], ignore_index=True)
    datasubs = data[[i for i in range(31)]]

    # Extract Features
    getF = getFeatures(datasubs.values)
    F1 = getF.fft()
    F2 = getF.entropy()
    F4 = getF.movingStd()
    F5 = getF.kurtosis()

    dataMatrix = np.concatenate((F1, F2[:, None], F4, F5[:, None]), axis=1)
    X = pd.DataFrame(data=dataMatrix)
    Y1 = data[31]
    selectedFeatures = [1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21, 30]  # Used Random forest to select best features

    # Apply K-fold on 3 models SVM, AdaBoost and Random Forest

    scorelabels = ['Accuracy', 'F1-score', 'Recall', 'Precision']

    data_new = np.concatenate((X[[i for i in selectedFeatures]], Y1[:, None]), axis=1)  # add labels
    execKfold(data_new, scorelabels)

    print('-------Using Random Forest-------------------')

    Y = data_new[:, -1]
    m = Models()
    m.fit('RandomForest', X[[i for i in selectedFeatures]], Y)

    joblib.dump(m, 'trainedModel.pkl')

    print('Success! \n Saved model to trainedModel.pkl')


if __name__ == '__main__':
    main()
