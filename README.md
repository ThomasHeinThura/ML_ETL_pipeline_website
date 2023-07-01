
## ETL CI/CD fraud detection website project

The end product is final website deployment.
* Using Scikit-learn, xgb-boost and many model to train. ( I want to add ensemble model)
* Prefect to orchestrate the whole project 
* MLFLow + Streamlit to show the trian model socre 
* Evidently + Streamlit to show the main report 
* Data profile? pandas_data_profiling?

# ----------------------------------------------------- #

### Streamlit website 

Take all the function and variable to build the website
report MLflow model accuracy in streamlit 
report evidently in streamlit 
can I do this ? if prefect is running show running
    
    
    #---------------------------------------------------------#
    |    model_select                  |                      |
    |    train, predict, custom, look  |  MLFlow model score  |
    |    custom -> select feature /    |       dataframe      |
    |               random gen values? |                      |
    |                                  |                      |
    |                                  |                      |
    |---------------#------------------#----------------------|
    |               |                                         |
    |  <T, P data>  |                                         |
    |  Pandas       |         Evidently report of             |
    |  Data         |      data shift and concept drift       |
    |  Profiling    |       classification and score          |
    |               |                                         |
    |               |                                         |
    |               |                                         |
    #---------------|-----------------------------------------#

# ----------------------------------------------------- #

There is three Database.
* This is for evidently database prograsql -> store report file
* MLflow database sql_lite -> store mlfile 
* Prefect database sql_lite -> store process file

# ----------------------------------------------------- #

Others thing that I like to add
* how about create the class that can call varible easily
* like data.train, data.valid_dataset, data.current_dataset
* Also testing the function and others. pylint black etc.
* Make file , setup and bash file. 
* CI/CD precommit 
* Docker and make file and also bash script.

# ----------------------------------------------------- #

