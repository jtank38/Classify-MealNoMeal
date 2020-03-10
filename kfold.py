from sklearn.model_selection import KFold
from models import *
import sys


def execKfold(featureMat):
    kf = KFold(20, True, 1)
    testmodel1 = []
    testmodel2 = []
    testmodel3 = []
    for train_index, test_index in kf.split(featureMat):
        train, test = featureMat[train_index], featureMat[test_index]
        X_train, Y_train,X_test,Y_test = train[:, :-1], train[:, -1],test[:, :-1], test[:, -1]
        models = kfoldTrain(X_train,Y_train)
        p1, p2, p3 = testing(models,X_test,Y_test)
        testmodel1.append(p1)
        testmodel2.append(p2)
        testmodel3.append(p3)

    testmodel1 = np.array(testmodel1)
    testmodel2 = np.array(testmodel2)
    testmodel3 = np.array(testmodel3)

    result1= np.mean(testmodel1,axis=0)
    result2= np.mean(testmodel2,axis=0)
    result3= np.mean(testmodel3,axis=0)

    resultlabels = ['accuracy','f1-score', 'recall','precision']
    for ind,i in enumerate(list(result1)):
        print('{0} is {1} for SVM \n'.format(resultlabels[ind],i))
    print('------------------------------------------------------')

    for ind,i in enumerate(list(result2)):
        print('{0} is {1} for AdaBoost \n'.format(resultlabels[ind],i))
    print('------------------------------------------------------')

    for ind,i in enumerate(list(result3)):
        print('{0} is {1} for randomForest \n'.format(resultlabels[ind],i))


def kfoldTrain(X_train,Y_train):

    m1 = Models()
    m1.fit('SVM',X_train,Y_train)
    m2 = Models()
    m2.fit('AdaBoost',X_train,Y_train)
    m3 = Models()
    m3.fit('RandomForest',X_train,Y_train)

    return [m1,m2,m3]


def testing(model,X_Test,Y_Test):
    testSVM = model[0].predict(X_Test,Y_Test)
    testAda = model[1].predict(X_Test, Y_Test)
    testRF = model[2].predict(X_Test, Y_Test)

    return testSVM,testAda,testRF
