import pandas as pd
import sqlite3

# Read the Excel file into a pandas DataFrame
df = pd.read_excel('Applications_for_Machine_Learning_internship_edited.xlsx')

# Connect to the SQLite database
conn = sqlite3.connect('sqlite.db')

# Create a new table in the database
table_name = 'applicants'
df.to_sql(table_name, conn, if_exists='replace', index=False)

# Commit the changes and close the database connection
conn.commit()
conn.close()