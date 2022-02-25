import sys,os
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV



from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score

from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


#sys.path.append(os.path.realpath('..'))

def load_data(database_filepath):
    """
    loads db file, creates engine and splits into features & labels
    
    Parameters
    ----------
    database_filepath: string
        file path relative to base directory of .db database
        
    Returns
    -------
    X: pandas DataFrame
        feature table containing ['message','original','genre'] columns
        
    Y: pandas DataFrame
        labels, 36 binary catgory columns
        
    category_names: list of strings
        36 names of categories, as ordered in Y
    
    """
    
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    df = pd.read_sql_table("DisasterResponse",engine)
    df.index=df['id']
    X = df[['message','original','genre']]
    Y = df.drop(columns=['id','message','original','genre'])
    
    category_names=list(Y.columns)
    
    return X,Y,category_names

def tokenize(text):
    """
    tokenizes text in message column for feature creation
    
    Parameters
    ----------
    
    text: string
        Un-edited message string from disaster related tweets
        
    
    Returns
    -------
    
    tokens: list of strings
        lemmatized token extracted from text, with stopwords removed
    """
    
    #lower case and add remove punctuation
    text=re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    
    #tokenize text
    tokens=word_tokenize(text)
    
    #remove stop words and lemmatize
    tokens = [WordNetLemmatizer().lemmatize(w)  for w in tokens if w not in stopwords.words("english")]
    
    return tokens


def build_model():
    """
    Contructs ML pipeline and creates grid search object
    
    returns
    -------
    
    model: GridSearchCV object
        grid search object, consisting of transform-evaulate pipeline and suitable parameter grid
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(lgb.LGBMClassifier()))
    ])
    
    parameters = {
            #'vect__ngram_range': ((1, 1), (1, 2)),
            'tfidf__use_idf': (True, False),
            #'clf__estimator__max_depth':[10,None]
        }
    
    model = GridSearchCV(pipeline,param_grid=parameters,cv=3)
    
    return model
    
    
    
    


def evaluate_model(model, X_test, Y_test, categories):
    """
    predicts on X_test using model, and displays evaluation metrics
    
    Parameters
    ----------
    
    model: Fitted GridSearchCV object
        fitted grid search object, consisting of transform-evaulate pipeline and suitable parameter grid. Fitted on X_train, Y_train
        
    X_test: pandas DataFrame
        test split of feature table containing ['message','original','genre'] columns
        
    Y_test: pandas DataFrame
        test split of labels, 36 binary catgory columns
        
    categories: list of strings
        36 names of categories, as ordered in Y_test
        
    """
    
    #make predictions
    Y_pred = model.predict(X_test['message'])
    
    #calculate confusion mat and accuracy
    accuracy = (Y_pred == Y_test).mean()
    try:
        confusion_mat = confusion_matrix(Y_test, Y_pred)


        #print classification report and confusion matrix for each label
        print("Confusio nMatrix:")
        for i,mat in enumerate(confusion_mat):
            print("#"*60)
            print(f"Classification report for category: { categories[i]}")
            print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))
            print(f"Confusion Matrix for category: {categories[i]}\n")
            sns.heatmap(mat,annot=True,fmt='g')
            plt.show()
    except:
        print("multilabel didn't work")
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    
    
    


def save_model(model, model_filepath):
    """
    saves model as pickle file in specified location
    
    Parameters
    ---------
    model: Fitted GridSearchCV object
        fitted grid search object, consisting of transform-evaulate pipeline and suitable parameter grid. Fitted on X_train, Y_train
        
    model_filepath: string
        location of pickle file to be saved
        
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    

    
    if len(sys.argv) == 3:
        
        nltk.download('stopwords')
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()