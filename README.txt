Final Full Notebook.ipynb:
Jupyter notebook.  Contains sample of code used to read data and build and evaluate models.  Model runs saved as pickle files.  Includes select visualizations as examples.

HelperFunctions.py:
All helper functions written over the course of project development. Contains code necessary for the sample notebook.

NOTE: In order to make our code fully runnable including web scraping and data creation, we modified it to avoid reading/writing to a database. In there are comments in the code describing the changes you would have to revert to in order to use a database.

FRB_H15.csv, Fundamentals.csv, GDP.csv:
csvs for all of our pulled data. Described in data section of final report.
These csvs were pulled from our database via queries like 
"SELECT * FROM [table name]"

They were uploaded by reading into pandas via read.csv and the "df.to_sql("[table name]") with the optional argument if_exists = replace. At the time when reuploading we chose to replace when we changed the data.

mainStockFrame.csv:
Dataframe we created from original datasets.

model_runs.p:
Pickle file of 100 random train test split trained models. LDA, Logistic Regression, and Random Forest.

model_runs_yearly.p:
Year based train test spilt trained final Random Forest Model
