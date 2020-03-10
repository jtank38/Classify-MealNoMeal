from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_selection import SelectFromModel
import sys


def RandomForest(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    # votedict = {}
    # for i in range(100):
    #     sel = SelectFromModel(RandomForestClassifier(n_estimators=200))
    #     sel.fit(X, Y)
    #     selected_feat = X.columns[(sel.get_support())]
    #     for i in selected_feat:
    #         if i not in votedict:
    #             votedict[i] = 1
    #         else:
    #             votedict[i] += 1
    # print(votedict)

    selectedFeatures = [1,2,3,4,5,6,17, 18, 19, 20, 21, 30]

    model = RandomForestClassifier()
    # model.fit(X_train[[i for i in selectedFeatures]], Y_train)
    #
    # rf_predictions = model.predict(X_test[[i for i in selectedFeatures]])
    #
    # rfc_cv_score = cross_val_score(model, X, Y, cv=20, scoring="roc_auc")
    # print(classification_report(Y_test, rf_predictions))
    # print('all scores is {}'.format(rfc_cv_score.mean()))

    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
    max_features = ["auto", "sqrt"]
    max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
    max_depth.append(None)
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth
    }
    rfc_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                    random_state=42, n_jobs=-1)

    rfc_random.fit(X_train[[i for i in selectedFeatures]], Y_train)
    print(rfc_random.best_params_)


def Adaptive(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada.fit(X_train, Y_train)
    ada_predicts = ada.predict(X_test)
    print(classification_report(Y_test, ada_predicts, output_dict=True)['1'])
    print(accuracy_score(Y_test, y_pred=ada_predicts, normalize=True))


def suppVM(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    modelSVM = SVC(kernel='rbf', gamma='scale')
    modelSVM.fit(X_train, Y_train)

    svm_predicts = modelSVM.predict(X_test)

    print(classification_report(Y_test, svm_predicts))
    print(accuracy_score(Y_test, svm_predicts, normalize=True))


class Models():
    def __init__(self):

        self.model = None

    def fit(self, modelName, X_train, Y_train):

        if modelName == 'SVM':
            self.model = SVC(kernel='rbf', gamma='scale')

        elif modelName == 'AdaBoost':
            self.model = AdaBoostClassifier(n_estimators=100, random_state=1)

        elif modelName == 'RandomForest':

            self.model = RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=None)  # 944 auto 140

        self.model.fit(X_train, Y_train)

    def predict(self, X_test, Y_test):

        prediction = self.model.predict(X_test)

        accuarcy = accuracy_score(Y_test, prediction, normalize=True)

        otherMetrics = classification_report(Y_test, prediction, output_dict=True)[
            '1.0']  # it's binary we can us any one

        return [accuarcy, otherMetrics['f1-score'], otherMetrics['recall'], otherMetrics['precision']]


if __name__ == '__main__':
    pass
