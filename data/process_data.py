import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads messages & categories csv files into pandas dataframes, and merges into a single dataframe
    
    Parameters
    ----------
    messages_filepath: string
        filepath of messages csb data
        
    categories_filepath: string
        filepath of categories csv data
    
    Returns
    -------
    df: pandas DataFrame 
    """
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    
    df=messages.merge(categories,how='left',left_on='id',right_on='id')
    return df


def clean_data(df):
    """
    Expands categories to columns with int values
    
    Parameters
    ----------
    df: pandas DataFrame
    
    Returns
    -------
    df_clean: pandas DataFrame 
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)    
    
    # select the first row of the categories dataframe
    row = categories. iloc[0] 

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x : x[0:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df` and create new dataframe 'df_clean'
    df_clean=df.drop(columns='categories')  
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_clean = pd.concat([df_clean,categories],axis=1)  
    
    # drop duplicates
    df_clean.drop_duplicates(inplace=True)
    
    return df_clean

def save_data(df, database_filename):
     """
     creates sqlite database with input name and saves pandas df to sqlite database
     
     Parameters
     ----------
     df: pandas DataFrame
        clean pandas data frame, ready for load to database
        
     database_filename: string
        name of sqlite database to be created
        
     Returns
     -------
     None
     
     """
    
    engine = create_engine(f'sqlite:///{database_filename}.db')
    df.to_sql(database_filename, engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()