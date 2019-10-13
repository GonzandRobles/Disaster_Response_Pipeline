import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories files
    and merge them into one dataset named df.
    input -> str, str
    output -> pandas DataFrame
    """
    # loads datasets paths
    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)

    # merge both datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    This function handles all the cleaning and
    column transformation of the datasets and returns
    a df with new column names without unnecesary strings,
    eliminates duplicates and concatenate the df with the
    new named columns.
    input -> DataFrame
    output -> DataFrame
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    # to create column names for the categories data.
    row = categories.iloc[0]
    category_colnames = (row.str.split('-').str)[0].tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1

    for column in categories:
        # set each value to be the last character of the string
        # and convert to integer datatype
        categories[column] = categories[column].str[-1:].astype('int')

    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True) # drop the original categories column from df
    df = pd.concat([df, categories], axis=1, join='inner') # concatenate the original dataframe with the new `categories` dataframe
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)
    return df

def save_data(df, database_filename):
    """
    save_data function saves the dataframe and
    turns it into a sql_database.
    input -> DataFrame
    output -> sql_database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')

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
