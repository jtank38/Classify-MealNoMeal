from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import tsfresh.feature_extraction.feature_calculators as tcal
import pandas as pd
import numpy as np
import joblib
np.set_printoptions(suppress=True)
import sys


class GetDirs:

    def missingValuesHelper(self, df, dfSeries):
        interpolated_data = []
        for series in dfSeries:
            cleaned_data = pd.Series(series).interpolate(method='linear', limit_direction='forward').to_list()
            interpolated_data.append(cleaned_data)

        return pd.DataFrame(interpolated_data, columns=df.columns)

    def getMealNoMealData(self, listFileNames):

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
            self.model = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=220)  # 944 auto 140

        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        prediction = self.model.predict(X_test)

        return [i for i in prediction]


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


def main():
    filename = input("[Note: If File in not under same directory please enter entire directory address] \n \nEnter the "
                     "Test File Name : ")

    # Read given CSV
    try:
        dataarr = GetDirs().getMealNoMealData(filename)

    except Exception as e:
        print("Couldn't find file under {0} Please Run and Try again!".format(filename))
        sys.exit(1)

    data = pd.DataFrame(data=dataarr)

    # Get Features
    getF = getFeatures(data.values)
    F1 = getF.fft()
    F2 = getF.entropy()
    F4 = getF.movingStd()
    F5 = getF.kurtosis()

    # Select feature subset
    selectedFeatures = [1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21, 30]

    X_test = np.concatenate((F1, F2[:, None], F4, F5[:, None]), axis=1)
    df_train = pd.DataFrame(data=X_test)

    # Load Model
    svm_from_joblib = joblib.load('trainedModel.pkl')

    # Run Prediction on Testset
    prediction = svm_from_joblib.predict(df_train[[i for i in selectedFeatures]])

    print('[Note: 1.0 is for Meal Data and 0.0 is for no Meal Data] \n Predictions are as follows:- \n \n')

    count = 0

    for i in prediction:
        print('Row {0} prediction is --> {1}'.format(count, i))
        count += 1


if __name__ == '__main__':
    main()
