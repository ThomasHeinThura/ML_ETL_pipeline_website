import psycopg2
import streamlit as st
from evidently.model_monitoring import ReportReader

# Connect to PostgreSQL
conn = psycopg2.connect(
    database="your_database",
    user="your_username",
    password="your_password",
    host="your_host",
    port="your_port"
)
cursor = conn.cursor()

# Retrieve report from Evidently AI
report_path = "path_to_evidently_report"
report_reader = ReportReader()
report = report_reader.read(report_path)

# Store report in PostgreSQL
table_name = "report_table"
create_table_query = """
    CREATE TABLE IF NOT EXISTS {} (
        id SERIAL PRIMARY KEY,
        report JSONB
    )
""".format(table_name)
cursor.execute(create_table_query)
insert_report_query = """
    INSERT INTO {} (report) VALUES (%s)
""".format(table_name)
cursor.execute(insert_report_query, (report,))
conn.commit()

# Display report using Streamlit
st.title("Evidently AI Report")
st.json(report)

# Close the database connection
cursor.close()
conn.close()
