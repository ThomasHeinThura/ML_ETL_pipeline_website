FROM python:3.10

WORKDIR ML_ETL_website

COPY requirements.txt .
COPY ./src ./src
COPY ./reports ./reports
COPY ./Data ./Data
COPY main.py . 
COPY Makefile .

RUN pip install -r requirements.txt

CMD [ "make", "run" ]
