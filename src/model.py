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
import xgboost as xgb





from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,confusion_matrix,classification_report


import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature

import logging

# connect MLFLOw

# class MODEL():
#     """
#     Return : report_ref
#              report_curr
#              model
#     """
    # def __init__():
    #     self.trian_data = train_data

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
        "Support Vector Classifier": SVC(),                             # Ridge, SVC, LinearSVC, Passive_AC
        "Decision Tree": DecisionTreeClassifier(max_depth=6),           #
        "KNearest": KNeighborsClassifier(n_neighbors=5),                # doesn't have model.predict_proba so I left out.
        "GaussianNB" : GaussianNB(),                                    #
        "LDA" : LinearDiscriminantAnalysis(),                           # 
        "Ridge" : RidgeClassifier(),                                    #  
        "QDA" : QuadraticDiscriminantAnalysis(),                        #
        "Bagging" : BaggingClassifier(),                                #
        "MLP" : MLPClassifier(),                                        #
        "LSVC" : LinearSVC(),                                           #  
        "BernoulliNB" : BernoulliNB(),                                  #  
        "Passive_AC" : PassiveAggressiveClassifier(),                   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
        "SGB"     : GradientBoostingClassifier(n_estimators=100, random_state=9),
        "Adaboost" : AdaBoostClassifier(n_estimators=100, random_state=9, algorithm='SAMME.R', learning_rate=0.8),
        "Extra_T" : ExtraTreesClassifier(n_estimators=100, max_features=3),
        "R_forest" : RandomForestClassifier(max_samples=0.9, n_estimators=100, max_features=3),
        "XGB" : xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),
        "Stacking" : stacking_model
    }
    return model 

# 

def eval_metrics(classifier, test_features, test_labels, avg_method):
    
    # make prediction
    predictions   = classifier.predict(test_features)
    # dataset["predictions"] = predictions
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
    
    target_names = ['0','1']
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(test_labels, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(f"{Matrix} \n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Base score: ", base_score)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)
    
    return base_score,accuracy,precision,recall,f1score,matrix_scores

def evaluate_model():
    """
    This is use to run for evaluate the model 
    Accuracy, classification report, confusion matrix, 
    f1, precision and recall
    """
    Models = build_model()  
    counter = 1
    for Model_Name, classifier in Models.items(): 
        # with mlflow.start_run(nested=True):
        print(f"{counter}. {Model_Name}")
        
        with mlflow.start_run():
            # fit the model
            classifier.fit(train_features, train_label)
            
            counter = counter + 1
            
            # Calculate the metrics
            base_score,accuracy,precision,recall,f1score,matrix_scores = eval_metrics(classifier,
                                                                                    test_features,
                                                                                    test_label, 
                                                                                    'weighted')  
            
            mlflow.log_param("Model"           , Model_Name)
            mlflow.log_metric("base_score"     , base_score)
            mlflow.log_metric("accuracy"       , accuracy)
            mlflow.log_metric("av_precision"   , precision)
            mlflow.log_metric("recall"         , recall)
            mlflow.log_metric("f1"             , f1score)
            mlflow.log_params(matrix_scores)
            
            signature = infer_signature(test_features, classifier.predict(test_features))
            if f1score > 0.945 :
                mlflow.sklearn.log_model(classifier,Model_Name, signature=signature)
                print(f"f1 socre is more than 0.945 so the {Model_Name} is saved")
            else :
                print(f"Because f1 socre is not quality. The model is skip to saving phase.")
            
            print("________________________________________")

    pass

def train_model_and_search_best_model():
    """
    Trian the models to search best model or 
    select model from website will be train
    """
    model = build_model() # may be if this in CLASS properties self.model is all I need
    # or 
    model = st.select() # select model from st.select
    train_data = get_data_to_train() # this will also self.train_data
    valid_data = get_data_to_valid_and_ref
    
    # train the model 
    for model_name, model in model:
        logging.info("Train the {model_name}")
        model.fit(train_data)
        
    # evaluate the model 
    evaluate_model(model, valid_data)
    
    # Also there is mlflow to search for best model .
    # train and save best model
    
    pass

def predict_the_model_and_report(
    # model=the_best_model
    ):
    """
    load the model and report the result through evidently
    """
    pass 

# return model, report_ref_data, report_current_data