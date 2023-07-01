
import pandas as pd
import sqlite3



import pandas as pd
import sqlite3



### Step if no DB create db 

# Class Databse -> train, ref(valid), cur(test) dataset, database
class Database():

    def __init__(self):
        self.train_data, self.valid_ref_data, self.current_data = self.make_data()
    
    def create_spl_DB_from_csv(self):
        """
        This will create DB to store raw data

        """
        # Connect to SQLite database
        conn = sqlite3.connect("Data/Fraud_database.db")
        
        # Read data from CSV file using pandas
        csv_file = "Data/Base.csv"
        df = pd.read_csv(csv_file)

        # Export DataFrame to SQLite database
        df.to_sql("Raw_data", conn, if_exists="replace", index=False)
        
        # clean the dataset
        df_clean = self.prepare_and_clean_dataset(df)
        
        # Export CLean Dataset to SQLite database
        df.to_sql("Clean_data", conn, if_exists="replace", index=False)
        
        conn.close()


    def prepare_and_clean_dataset(self, dataset):
        """
        This will clean the dataset 
        """
        categorical_columns = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder


        # Apply label encoding to the categorical column
        le = LabelEncoder()
        for col in categorical_columns:
            dataset[col] = le.fit_transform(dataset[col])

        return dataset 
    
# ----------------------------------------------------- #
    def get_data_from_database(self):
        """
        this is the test hypothesis where 
        train, valid and current dataset become overlap
        then create main big file and split that
        where fraud_bool 0 and 1 is equal and random
        """
        
        conn = sqlite3.connect("Data/Fraud_database.db")
        # Create a cursor object
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Clean_data'")
        table_exists = cursor.fetchone()
        sample_size = 11_000
        query  = None

        if not table_exists:
            # If the table doesn't exist, call the create_sql_dataset function
            self.create_spl_DB_from_csv()
            
        else:
            # Retrieve data from the database
            query = f"""
                SELECT *
                FROM Clean_data
                WHERE fraud_bool = ?
            """

        fraud_true = pd.read_sql_query(query, conn, params=[True]).sample(n=sample_size, random_state=42).reset_index(drop=True)
        fraud_false = pd.read_sql_query(query, conn, params=[False]).sample(n=sample_size, random_state=42).reset_index(drop=True)

        # Close the database connection
        conn.close()
            
        return fraud_false, fraud_true
        
    def make_data(self):
        """
        Get data from DB to tain the model
        take : sql database(50% of database)
        output : df
        
        
        main data
            |
        #------------#
        |            |
    train     ref_current_data
                    |
              #-------------#
              |             |
           ref_Data      cur_Data
    
        """
        
        fraud_false, fraud_true = self.get_data_from_database()
        
        from sklearn.model_selection import train_test_split
        
        fraud_dataset = pd.concat((fraud_true, fraud_false), axis=0)
        
        train_data, ref_current_data = train_test_split(fraud_dataset, test_size = 0.5, random_state = 42)
        
        ref_data, current_data = train_test_split(ref_current_data, test_size = 0.5, random_state = 42)
        
        
        return train_data, ref_data, current_data
    
# ----------------------------------------------------- #
