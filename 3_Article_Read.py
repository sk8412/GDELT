# code adapted from: https://github.com/walfaelschung/GDELT_flow#readme

# Imports
from newspaper import Article
import pandas as pd

# Initializing variables
count = 0
count_fail = 0

# Loading .csv of URLs to follow
file_to_read = "results_tagged_v1.csv"
# url = pd.read_csv(file_to_read, sep=',', header = 0, usecols=['GlobalEventID'])
url = pd.read_csv(file_to_read, sep=',', header = 0, usecols = ['GlobalEventID', 'SOURCEURL', 'match30', 'match50', 'match70'])

# Setting the column names to be used in the dataframe
colnames = ['GlobalEventID', 'SOURCEURL', 'Title', 'Text', 'match30', 'match50', 'match70']

# Initializing the dataframes
title_df = pd.DataFrame(columns=colnames)
articles_df = pd.DataFrame(columns=colnames)

# Iterating through list of imported URLs
for index, row in url.iterrows():
    article = Article(row['SOURCEURL'])
    article.download()
    
    # Checks if download was successful
    # Parses through downloaded article
    # Pulls title and text from article
    if article.download_state == 2:     
        article.parse()
        title = article.title
        text = article.text

        # new_title consists of each title and text pulled from the article
        # It is concatenated with articles_df, which keeps a list of all titles and texts
        new_title = pd.DataFrame([[row['GlobalEventID'], row['SOURCEURL'], title, text, \
                                   row['match30'], row['match50'], row['match70'], ]], columns=colnames)
        articles_df = pd.concat([articles_df, new_title], ignore_index=True)
        #print(title)
        count += 1
        print('Just completed importing article number:', count)
    else:
        #print(f'{row['SOURCEURL']} failed')
        count_fail += 1

# Sends title and text of each article to a .csv
articles_df.to_csv('articles_labeled.csv',index=False, sep=',', )
print('Number of articles sucessfully imported to the csv:', count)
print('Number of broken URLs:', count_fail)