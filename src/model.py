# import sklearn model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import  RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,confusion_matrix,classification_report


import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature

import logging

def build_model():
    """
    Building multi model to trian and search for best model 
    """
    from sklearn.ensemble import StackingClassifier

    base_models=[('RF',RandomForestClassifier(max_samples=0.9,n_jobs=-1)),('knn',KNeighborsClassifier(n_neighbors=5,n_jobs=-1))]
    meta_model = LogisticRegression(n_jobs=-1)
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=True, cv=3,n_jobs=-1)

    model = {
        "Logistic Regression": LogisticRegression(),                    #
        # "Support Vector Classifier": SVC(),                           # Ridge, SVC, LinearSVC, Passive_AC
        "Decision Tree": DecisionTreeClassifier(max_depth=6),           #
        "KNearest": KNeighborsClassifier(n_neighbors=5),                # doesn't have model.predict_proba so I left out.
        "GaussianNB" : GaussianNB(),                                    #
        "LDA" : LinearDiscriminantAnalysis(),                           # 
        # "Ridge" : RidgeClassifier(),                                  #  
        "QDA" : QuadraticDiscriminantAnalysis(),                        #
        "Bagging" : BaggingClassifier(),                                #
        "MLP" : MLPClassifier(),                                        #
        # "LSVC" : LinearSVC(),                                         #  
        "BernoulliNB" : BernoulliNB(),                                  #  
        # "Passive_AC" : PassiveAggressiveClassifier(),                 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
        "SGB"     : GradientBoostingClassifier(n_estimators=100, random_state=9),
        "Adaboost" : AdaBoostClassifier(n_estimators=100, random_state=9, algorithm='SAMME.R', learning_rate=0.8),
        "Extra_T" : ExtraTreesClassifier(n_estimators=100, max_features=3),
        "R_forest" : RandomForestClassifier(max_samples=0.9, n_estimators=100, max_features=3),
        # "XGB" : xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05), # cannot train
        "Stacking" : stacking_model
    }
    return model 

# 

def eval_metrics(classifier, test_features, test_labels, avg_method):
    
    # make prediction
    predictions   = classifier.predict(test_features)
    base_score   = classifier.score(test_features,test_labels)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average=avg_method)
    recall = recall_score(test_labels, predictions, average=avg_method)
    f1score = f1_score(test_labels, predictions, average=avg_method)
    Matrix = confusion_matrix(test_labels, predictions)
    matrix_scores = { 
        "true negative"  : Matrix[0][0],
        "false positive" : Matrix[0][1],
        "false negative" : Matrix[1][0],
        "true positive " : Matrix[1][1]
    }
    
    # uncomment for debugging
    
    # target_names = ['0','1']
    # print("Classification report")
    # print("---------------------","\n")
    # print(classification_report(test_labels, predictions,target_names=target_names),"\n")
    # print("Confusion Matrix")
    # print("---------------------","\n")
    # print(f"{Matrix} \n")

    # print("Accuracy Measures")
    # print("---------------------","\n")
    # print("Base score: ", base_score)
    # print("Accuracy: ", accuracy)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1 Score: ", f1score)

    return base_score,accuracy,precision,recall,f1score,matrix_scores
