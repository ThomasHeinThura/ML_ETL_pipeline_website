# command for mlflow prefect and setupfile

run :
	# run mlflow add docker network as host `--host 172.18.0.2:500` to show in docker
	mlflow server --backend-store-uri sqlite:///reports/backend.db &

	# run streamlit 
	streamlit run main.py

setup:
	#install python library
	pip install -r requirements.txt

docker_build :
	#Build Docker
	sudo docker build -t ml_etl_website --network loki .
	

docker_run:
	#Run Docker
	sudo docker run --network asgard -p 5000:5000 -p 8501:8501 --name ml_etl ml_etl_website

docker_start:
	sudo docker start ml_etl


