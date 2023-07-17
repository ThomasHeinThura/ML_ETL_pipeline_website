import sqlite3
import mlflow
import streamlit as st

# Connect to MLflow SQLite backend
conn = sqlite3.connect("sqlite:///backend.db")

# Retrieve model value from MLflow
experiment_id = "your_experiment_id"
run_id = "your_run_id"
model_key = "your_model_key"

query = """
    SELECT value
    FROM artifacts
    WHERE run_id = ?
        AND key = ?
"""

cursor = conn.cursor()
cursor.execute(query, (run_id, model_key))
result = cursor.fetchone()

if result is not None:
    model_value = result[0]
else:
    st.error("Model value not found in the database.")

# Display model value using Streamlit
st.title("MLflow Model Value")
st.code(model_value)

# Close the database connection
cursor.close()
conn.close()
