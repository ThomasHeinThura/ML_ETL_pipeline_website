### Import necessary library

import logging
import sys
sys.path.append("src")

from streamlit_function import train_button_action
from streamlit_function import show_on_pandas_profiling
from streamlit_function import show_on_mlflow_section
from streamlit_function import show_on_evidently_section

from model import build_model


### Streamlit website 

    # Take all the function and variable to build the website
    # report MLflow model accuracy in streamlit 
    # report evidently in streamlit 
    # can I do this ? if prefect is running show running
    
    
    # |---------------------------------------------------------|
    # |    model_select                  |                      |
    # |    train, predict, custom, look  |  MLFlow model score  |
    # |    custom -> select feature /    |       dataframe      |
    # |               random gen values? |                      |
    # |                                  |                      |
    # |                                  |                      |
    # |---------------|------------------|----------------------|
    # |               |                                         |
    # |  <T, P data>  |                                         |
    # |  Pandas       |         Evidently report of             |
    # |  Data         |      data shift and concept drift       |
    # |  Profiling    |       classification and score          |
    # |               |                                         |
    # |               |                                         |
    # |               |                                         |
    # |---------------|-----------------------------------------|


import streamlit as st 

# page config with wide
st.set_page_config(page_title="ETL", page_icon="🌐", layout="wide")

st.title("ELT CI/CD fraud dection website")
st.markdown(
    """
<style>
    # [data-testid="collapsedControl"] {display: none}
    MainMenu {visibility : hidden;}
    footer {visibility : hidden;}
    header {visibility : hidden}
</style>
    """,unsafe_allow_html=True,
)
# ----------------------------------------------------- #

# basic config
model = build_model()
train_button = None
model_select = None 
ref_dataset = None
current_dataset = None
return_model = None


with st.container():
    right_col, left_col = st.columns([1,2])
    
    with right_col: # Train and pandas profilingS
        with st.container(): #Select model to train
            model_select = st.selectbox(
                "Select model to train",
                (model.keys())
                )
        # ----------------------------------------------------- #
        with st.container(): # Train model
            train_button = st.button("Train the model")
            if train_button:
                return_model = train_button_action(model[model_select])

        # ----------------------------------------------------- #
        with st.container(): # pandas profiling
            show_on_pandas_profiling()
                
# ----------------------------------------------------- #
    with left_col: # MFlow model view section
        if train_button:
            # st.write("---")
            pass
        with st.container():
            st.header("MLFlow Models and Scores")
            show_on_mlflow_section()
# ----------------------------------------------------- #                    

with st.container(): # Evidently report
    st.write("---")
    st.header("Evidently AI report") 
    with st.container():
        if train_button:
            show_on_evidently_section(return_model)