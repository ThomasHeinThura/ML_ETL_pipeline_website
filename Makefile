# command for mlflow prefect and setupfile

run :
	# run mlflow 
	mlflow server --backend-store-uri sqlite:///reports/backend.db --host 127.0.0.1:5000 &

	# run streamlit 
	streamlit run main.py

setup:
	#install python library
	pip install -r requirements.txt