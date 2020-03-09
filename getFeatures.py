import tsfresh.feature_extraction.feature_calculators as tcal
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
        return np.array([tcal.kurtosis(self.data[i,:]) for i in range(len(self.data))])


class getPCA():
    def __init__(self, featureMat):
        self.mat = featureMat
        self.normData=StandardScaler().fit_transform(self.mat)
        self.pca_timeSeries=PCA(n_components=5)
        self.PCs = self.pca_timeSeries.fit_transform(self.normData)
        self.df = pd.DataFrame(self.PCs, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

    def pca(self):

        return self.df.values

    def results(self):

        return self.pca_timeSeries.explained_variance_ratio_


class plotDiag():
    def __init__(self,title, color, data, xLabel, yLabel,FileName):
        self.title=title
        self.color=color
        self.input= data
        self.xLabel=xLabel
        self.yLabel = yLabel
        self.File = FileName

    def plot(self):
        plt.title(self.title)
        plt.xlabel(self.xLabel)
        plt.ylabel(self.yLabel)
        plt.scatter(range(len(self.input)),self.input, c=self.color)
        plt.grid(True)
        plt.savefig(self.File)
        plt.show()

if __name__ == '__main__':
    a = getFeatures()
