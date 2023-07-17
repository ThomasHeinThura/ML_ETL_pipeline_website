
import streamlit as st
from dataset_prepare import Database

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components

from model import eval_metrics
from model import evaluate_model

from os.path import exists



# ----------------------------------------------------- #

Database_website = Database()

label = ['fraud_bool']

# Streamlit function 
def train_button_action(model,):
    """
    This action will train the dataset 
    and show pandas profiling and 
    according to the model selection
    """
    st.write("This is train_data")
    st.write(f"The train model is {model}")
    # st.dataframe(Database_website.train_data)
    # Database_website.train_data
    
    # train the model 
    with st.spinner('Model is trian'):
        model.fit(
        Database_website.train_data.drop(label, axis=1).to_numpy(),
        Database_website.train_data[label].to_numpy())
    
    # evaluate the model 
    with st.spinner("Evaluate the model with ref data"):
        # pred = model.predict(Database_website.valid_ref_data.drop(label, axis=1).to_numpy())
        
        base_score   = model.score(
            Database_website.valid_ref_data.drop(label, axis=1).to_numpy(),
            Database_website.valid_ref_data[label])
        
        eval_metrics(
            model, 
            Database_website.valid_ref_data.drop(label, axis=1).to_numpy(),
            Database_website.valid_ref_data[label],
            'weighted'
                     )
        
        st.write(f'The base score is : {base_score}')
    
    # add data to mlflow  sql and show
    
    # show_on_mlflow_section()



# This section is removed bz predict model need to save model

# def predict_button_action(model,):
#     """
#     This action will predict the dataset
#     according to the model selection
#     """
    
#     # load_predict_model = model
#     st.write("This is Test and current data")
#     st.write(f'The predict model is {model}')

#     # st.dataframe(Database_website.current_data)
    
#     # base_score = model.score(
#     #     Database_website.current_data.drop(label, axis=1).to_numpy(),
#     #     Database_website.current_data[label]
#     # )
    
#     # st.write(f"The base score is : {base_score}")
    
#     pass
    

# This section is also removed bz it takes a lot time to adjust

# def custom_button_action():
#     """
#     This will allow to fill custom data
#     and predict using custom data
#     """
#     st.write("This is custom data")
#     with st.spinner("Train all model"):
#         evaluate_model()
#     pass

# left column
def show_on_mlflow_section():
    """
    This will show on mlflow section
    take data from MLFlow sql
    """
    
    components.iframe("http://127.0.0.1:5000/", width=1350, height=800, scrolling=True)
    pass

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
        height = 600
        
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
            

        else :
            st.write("Select the dataset")
        
    
    

# left column
def show_on_evidently_section():
    """
    This will show on evidently section
    """
    pass 

# ----------------------------------------------------- #