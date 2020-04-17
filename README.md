# DisasterResponsePipeline
Web app project with ETL, NLP and ML pipelines to handle real messages that were sent during disaster events with the purpose to be sent to an appropriate disaster relief agency. 

# Project Overview
This project is part of the Data Science NanoDegree from Udacity. With data from Figure Eight, the main goal is to analyze disaster data to build a model for an API that classifies disaster messages. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

# Files
* process_data.py -> This file includes the code that loads the data, cleans the data and saves the data into a sql_database.
* train_classifier.py -> This file loads and split data into training X,y sets, it also builds a nlp pipeline and uses a gridsearch object to evaluate the model.
* custom_transformer.py -> This file contains the tokenize function that will handle the text processing of the messages, and it also contains the custom made transformer class TextLength for creating a text length feature to be used in the ML pipeline. 
* run.py -> This file has the flask web-app code that loads the model, display the visuals and receive user input text for modeling.
* categories.csv -> categories data
* messages.csv -> messages data

# Dependencies
- Python 3.5+ (I used Python 3.7)
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Web App and Data Visualization: Flask, Plotly

# Installing
Clone this GIT repository [https://github.com/GonzandRobles/Disaster_Response_Pipeline.git](here)

**Executing Program:**
1. Run the following commands in the project's root directory to set up your database and model.
  * To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  * To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  
2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/
