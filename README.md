# Disaster Response Pipeline Project

**Background:** Message data from Figure 8 has been collated from real tweets/texts that were sent during the time of disaster events. These are the communications response workers will recieve as the disaster. However, due to their large volume and varied nature, it is difficult for responders to use these communications to aid rescue/ relief in the often tight time windows. Often, only 1 in 1000 messages ent at the time are relevent to the response effort.

The messages in the Figure 8 dataset have been labelled into 36 categories, to make it easier to quickly determine the message's relevance to the disaster response processionals. Using these pre-labelled messages, a classifier can be trained that will automatically sort messages into categories relevent to disaster response professionals.

This classifier can be used in a a web app, to allow emergency workers to input messages and quickly get classifications. This will aid in reducing the time spent reviewing messages.

## Instructions:
Follow instructions to intiate database, train model and launch web app

0. Prequisite, pip install lightgbm, if not already installed

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory (cd app) to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to view api

## Repo Contents:

1. app
    - templates (html templates for web app)
        - go.html
        - master.html
    - run.py (run script to launch web app)
        'Script rangles disaster db and produces visuals, as well as categorizing unseen messages into disaster categories'
    
2. data
    - disaster_categories.csv
        'Columns: id, categories (list of categories)'
    - disaster_messages.csv
        'Columns: id, message, original, genre'
    - process_data.py
        'Loads csv disaster sources, cleans and saves to sql db'
        
3. models
    - train_classifier.csv
        'Script that transforms and fits a classifier onto disaster response data. Includes parameter tuning with gridsearch CV'
