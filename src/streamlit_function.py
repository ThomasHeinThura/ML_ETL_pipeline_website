
import streamlit as st
from dataset_prepare import Database

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components

from model import eval_metrics
from model import evaluate_model



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


def predict_button_action(model,):
    """
    This action will predict the dataset
    according to the model selection
    """
    
    # load_predict_model = model
    st.write("This is Test and current data")
    st.write(f'The predict model is {model}')

    # st.dataframe(Database_website.current_data)
    
    # base_score = model.score(
    #     Database_website.current_data.drop(label, axis=1).to_numpy(),
    #     Database_website.current_data[label]
    # )
    
    # st.write(f"The base score is : {base_score}")
    
    pass
    

def custom_button_action():
    """
    This will allow to fill custom data
    and predict using custom data
    """
    st.write("This is custom data")
    with st.spinner("Train all model"):
        evaluate_model()
    pass

# left column
def show_on_mlflow_section():
    """
    This will show on mlflow section
    take data from MLFlow sql
    """
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
    start_button = st.button("Start profiling")
    if start_button:
        with st.spinner("Start profiling it might take some time"):
            if selected_box == "train_data":
                report_path = "Train_data.html"
                with open(report_path, encoding="utf8") as report_f:
                    report: Text = report_f.read()
                    components.html(report, width=1000, height=1200, scrolling=True)
                # profile_report = Database_website.train_data.profile_report()

                # st_profile_report(profile_report)
            elif selected_box == "valid_ref_data":
                profile_report = Database_website.valid_ref_data.profile_report()
                # profile_report.to_file(f"Train_data.html")
                st_profile_report(profile_report)
            elif selected_box == "current_data":
                profile_report = Database_website.current_data.profile_report()
                # profile_report.to_file(f"Train_data.html")
                st_profile_report(profile_report)
            else :
                st.write("Select the dataset")
        
    
    

    
    
    

# left column
def show_on_evidently_section():
    """
    This will show on evidently section
    """
    pass 

def test_the_model_and_dataset():
    """
    Test the model and dataset through evidently by prefect
    """
    pass

# ----------------------------------------------------- #