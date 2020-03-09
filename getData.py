import pandas as pd
import numpy as np
import os



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


if __name__ == '__main__':
    a = GetDirs()
    f1 = a.getDirs('MealData')
    f2 = a.getDirs('NoMealData')
    meal = np.array([])
    noMeal = np.array([])

    meal = np.concatenate([a.getMealNoMealData(i,'Meal') for i in f1])
    noMeal = np.concatenate([a.getMealNoMealData(i) for i in f2])

    # meal = a.getMealNoMealData(f1,'Meal')
    # noMeal = a.getMealNoMealData(f1)
    print(meal.shape,noMeal.shape)
