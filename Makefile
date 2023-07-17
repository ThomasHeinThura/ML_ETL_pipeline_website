# command for mlflow prefect and setupfile

run :
	# run mlflow 
	mlflow server --backend-store-uri sqlite:///reports/backend.db &

	# run streamlit 
	streamlit run main.py

setup:
	#install python library
	pip install -r requirements.txt