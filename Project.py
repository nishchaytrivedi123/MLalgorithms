import numpy as np
import pandas as pd
from missingpy import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# take user input to let him select for which year's data he wants to take
datasetyear = input("Which year dataset you want to take ( from 1-5 or to get dataset of combined all years enter 0)")
datasetyear = int(datasetyear)

# based on the user input load file
if datasetyear == 0:
    data = pd.read_csv('all_years.csv')

if datasetyear == 1:
    data = pd.read_csv('1year.csv')    

if datasetyear == 2:
    data = pd.read_csv('2year.csv')

if datasetyear == 3:
    data = pd.read_csv('3year.csv')

if datasetyear == 4:
    data = pd.read_csv('4year.csv')

if datasetyear == 5:
    data = pd.read_csv('5year.csv')

# ask user to choose value for k cross validation (5 or 10)
k_fold = input("Enter k value for k-fold cross validation(5 or 10): ")
k_fold = int(k_fold)

#  let user selects which kernel to be chosen for SVM
svm_kernel = input("For SVM which kernel you want to choose? (for polynomial enter 'poly', for linear enter 'linear', for radial enter ' rbf')")

# taking the target values in one dataframe from all dataset
dt = data[data.columns[len(data.columns)-1]]
target = dt

# the data without target values
data = data[data.columns[:64]]

# standard scalar(z-score) used for feature selection 
scaler = StandardScaler()
scaled_df = scaler.fit_transform(data)

# for knn imputation
imputer = KNNImputer(n_neighbors=5, weights="distance")
imputed_data = imputer.fit_transform(scaled_df)

# used stratified k fold cross validation
skf = StratifiedKFold(n_splits=k_fold)

# SVM
print("SVM")

svm_acc = 0
svm_spe = 0
svm_sen = 0

# k fold cross validation loop
for train, test in skf.split(imputed_data,target):

    # divide dataser in training and testing
    X_train, X_test, y_train, y_test = imputed_data[train] , imputed_data[test], target[train], target[test]
    
    # smote used to balance the data
    sm = SMOTE()

    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    # clf = svm.SVC(gamma='auto', kernel='linear') # for linear kernel
    # clf = svm.SVC(gamma='auto', kernel='rbf') # for radial kernel

    # SVM model is used
    clf = svm.SVC(gamma='auto', kernel=svm_kernel)
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)

    # check the accuracy score
    svm_acc += metrics.accuracy_score(y_test, y_pred)

    # confusion matrix result
    tp, fn, fp, tn = confusion_matrix(y_test,y_pred).ravel()

    # check the specificity score
    specificity = (tn/(tn+fp))
    svm_spe += specificity

    # check the sensitivity score
    sensitivity = (tp/(tp+fn))
    svm_sen += sensitivity

# show the results o console 
print("Accuracy:", svm_acc/k_fold )
print("Specificty:", svm_spe/k_fold )
print("Sensitivity:", svm_sen/k_fold )

# Random Forest
print("RANDOM FOREST")

# for hyper parameter tuning
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)]
max_features = ['sqrt']
bootstrap = [True]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'bootstrap': bootstrap}

rf_acc = 0
rf_spe = 0
rf_sen = 0

# k fold loop
for train, test in skf.split(imputed_data, target):
    # divide dataser in training and testing
    X_train, X_test, y_train, y_test = imputed_data[train] , imputed_data[test], target[train], target[test]
    
    # smote used
    sm = SMOTE()

    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    # random forest classifier
    model = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train_res, y_train_res)
    parameters = list(rf_random.best_params_.values())

    # parameters are chosen after performing hyper parameter tuning
    model_random = RandomForestClassifier(n_estimators=parameters[0], bootstrap = parameters[2], max_features = parameters[1])
    model_random.fit(X_train_res, y_train_res)
    
    # predictinng the result
    y_pred=model_random.predict(X_test)

    # calculate accuracy score
    rf_acc += metrics.accuracy_score(y_test, y_pred)

    # get confusion matrix
    tp, fn, fp, tn = confusion_matrix(y_test,y_pred).ravel()

    # calculate specificity score
    specificity = (tn/(tn+fp))
    rf_spe += specificity

    # calculate sensitivity score
    sensitivity = (tp/(tp+fn))
    rf_sen += sensitivity


# show the results on console
print("Accuracy:", rf_acc/k_fold )
print("Specificty:", rf_spe/k_fold )
print("Sensitivity:", rf_sen/k_fold )

# Naive Bayes
print("Naive Bayes")

nb_acc = 0
nb_spe = 0
nb_sen = 0

# k fold
for train, test in skf.split(imputed_data, target):

    # split  data in training and testing
    X_train, X_test, y_train, y_test = imputed_data[train] , imputed_data[test], target[train], target[test]
    
    # smote is used
    sm = SMOTE()

    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
   
    # gaussian versionn of naive bayes is used because of all continuous features
    model = GaussianNB()
    model.fit(X_train_res, y_train_res)

    # predict the result
    y_pred=model.predict(X_test)

    # calculate accuracy score
    nb_acc += metrics.accuracy_score(y_test, y_pred)

    # confusion matrix result
    tp, fn, fp, tn = confusion_matrix(y_test,y_pred).ravel()

    # calculate specificity score
    specificity = (tn/(tn+fp))
    nb_spe += specificity

    # calculate sensitivity score
    sensitivity = (tp/(tp+fn))
    nb_sen += sensitivity

# shows the result on console
print("Accuracy:", nb_acc/k_fold )
print("Specificty:", nb_spe/k_fold )
print("Sensitivity:", nb_sen/k_fold )