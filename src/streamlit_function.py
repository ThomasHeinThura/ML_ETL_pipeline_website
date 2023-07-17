
import streamlit as st
from dataset_prepare import Database

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components

from model import eval_metrics

from os.path import exists

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature


from data_monitor import fit_and_save_report

# ----------------------------------------------------- #

Database_website = Database()

label = ['fraud_bool']

# Streamlit function 
def train_button_action(model):
    """
    This action will train the dataset 
    and show pandas profiling and 
    according to the model selection
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(nested=True):
        # train the model 
        with st.spinner('Model is trian'):
            model.fit(
            Database_website.train_data.drop(label, axis=1).to_numpy(),
            Database_website.train_data[label].to_numpy())
        
        # evaluate the model 
        with st.spinner("Evaluate the model with ref data"):
            base_score,accuracy,precision,recall,f1score,matrix_scores = eval_metrics(
                model, 
                Database_website.valid_ref_data.drop(label, axis=1).to_numpy(),
                Database_website.valid_ref_data[label],
                'weighted')

            mlflow.log_param("Model"           , model)
            mlflow.log_metric("base_score"     , base_score)
            mlflow.log_metric("accuracy"       , accuracy)
            mlflow.log_metric("av_precision"   , precision)
            mlflow.log_metric("recall"         , recall)
            mlflow.log_metric("f1"             , f1score)
            mlflow.log_params(matrix_scores)
            
            return model

# ----------------------------------------------------- #

# left column
def show_on_mlflow_section():
    """
    This will show on mlflow section
    take data from MLFlow sql
    """
    components.iframe("http://127.0.0.1:5000/", width=1350, height=800, scrolling=True)
# ----------------------------------------------------- #

# right column
def show_on_pandas_profiling():
    """
    show in pandas profiling 
    from select data(train, predict, custom)
    """
    selected_box = st.selectbox(
        "Select Data to show profiloing",
        ("train_data",
         "valid_ref_data",
         "current_data")
    )

    with st.spinner("Start profiling it might take some time"):
        width = 650
        height = 650
        
        if selected_box == "train_data":
            # if there is no profile report create profile report 
            # if there is profile report use profile report 
            report_path = "reports/pandas/train_data.html"
            if  exists(report_path) == True:
                with open(report_path, encoding="utf8") as report_f:
                    report: Text = report_f.read()
                    components.html(report, width=width, height=height, scrolling=True)
            else:
                profile_report = Database_website.train_data.profile_report()
                profile_report.to_file(f"reports/pandas/train_data.html")
                st_profile_report(profile_report)
        
        elif selected_box == "valid_ref_data":
            report_path = "reports/pandas/valid_data.html"
            if  exists(report_path) == True:
                with open(report_path, encoding="utf8") as report_f:
                    report: Text = report_f.read()
                    components.html(report, width=width, height=height, scrolling=True)
            else:
                profile_report = Database_website.valid_ref_data.profile_report()
                profile_report.to_file(f"reports/pandas/valid_data.html")
                st_profile_report(profile_report)
            
        elif selected_box == "current_data":
            report_path = "reports/pandas/current_data.html"
            if  exists(report_path) == True:
                with open(report_path, encoding="utf8") as report_f:
                    report: Text = report_f.read()
                    components.html(report, width=width, height=height, scrolling=True)
            else:
                profile_report = Database_website.current_data.profile_report()
                profile_report.to_file(f"reports/pandas/current_data.html")
                st_profile_report(profile_report)
        
# ----------------------------------------------------- #

# left column
def show_on_evidently_section(return_model):
    """
    This will show on evidently section
    """
    st.write(f"Models check for data drift : {return_model}")
    fit_and_save_report(return_model, Database_website.valid_ref_data, Database_website.current_data, label)
    
    
    
    
    pass 

# ----------------------------------------------------- #