# run mlflow 
mlflow server --backend-store-uri sqlite:///reports/backend.db

# run streamlit 
streamlit run main.py