### Import necessary library

import logging
import sys
sys.path.append("src")

from streamlit_function import train_button_action
from streamlit_function import predict_button_action
from streamlit_function import custom_button_action
from streamlit_function import show_on_pandas_profiling

from model import build_model


# import pandas 
# import numpy
# import evidently

# import prefect 
# import os 

# log config

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

model = build_model()

train_button = None
predict_button = None 
custom_button = None
model_select = None 
ref_dataset = None
current_dataset = None

with st.container():
    right_col, left_col = st.columns([1,2])
    
    
    with right_col: # Button and Action section
        with st.container():
            first, second, third = st.columns(3)
            model_select = st.selectbox(
                "Select model to train",
                (model.keys()))
            
            st.markdown("The prefect markdown to link")
            
            with st.container():
                with first:
                    train_button = st.button("Train the model")
                    if train_button:
                        train_button_action(model[model_select])
                        # st.write(f'The trian model is {model_select}')
                        # st.write(f'The classifier is {model[model_select]}')
                        pass
                
                with second:
                    predict_button = st.button("Predict with the best model")
                    if predict_button:
                        predict_button_action(model[model_select])
                        st.write(f'The predict model is {model_select}')
                        pass
                
                with third:
                    custom_button = st.button("Train all model")
                    if custom_button:
                        custom_button_action()
                        pass

    with left_col: # MFlow model view section
        with st.container():
            st.markdown("This is for MLflow model best result")
            if train_button:
                with st.spinner("Loading and Updating"):
                    st.write("Model update")

with st.container():
    st.write("---")
    right_col, left_col = st.columns([1,2])

    with right_col: # Pandas Profiling section
        with st.container(): 
            show_on_pandas_profiling()
                
            # if train_button: 
            #     st.write("This is Train Data")
                
                
            # elif predict_button:
            #     st.write("This is Predict Data")
                
            # elif custom_button:
            #     st.write("This is custom Data")
                
            # else: 
            #     st.write("This is for pandas profiling")
            
    
    with left_col: # Evidently section
        with st.container():
            # st.selectbox("Report viewer")
            st.write("This is Evidently report")
            if predict_button:
                st.write("Relaod Evidently report")