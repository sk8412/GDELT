# code adapted from: https://github.com/walfaelschung/GDELT_flow#readme

# Imports
from newspaper import Article
from datetime import datetime
import pandas as pd
import time
import os


# Initializing variables
count_success = 0
count_fail = 0
row_count = 0
first_row = 1

# Rows per file
CHUNK_SIZE = 1000

# Report frequency
PRINT_FREQ = 10

# Data file name
FN_DATA = "/home/skyler/results_tagged_v1.csv"

# Output file name
FN_OUT = 'articles_{:09d}_{:09d}_{}.csv'

# Output data folder
DF_OUT = '/data2/skyler/gdelt_v3'

url = pd.read_csv(FN_DATA, sep=",", header=0,
                  usecols=["GlobalEventID", "SOURCEURL",
                           "match30", "match50", "match70"])

# Total number of rows to be analyzed
total_count = str(url.shape[0])

# Setting the column names to be used in the dataframe
colnames = ["GlobalEventID", "SOURCEURL", "Title", "Text",
            "match30", "match50", "match70"]

# Initialize results list
df_list = []

# Start timer
start_time = time.time()

# Iterating through list of imported URLs
for index, row in url.iterrows():
    article = Article(row["SOURCEURL"])
    article.download()
    row_count += 1

    if row_count % PRINT_FREQ == 0:
        print('Importing article number ' +
              str(row_count) + ' of ' + str(total_count))

    # Checks if download was successful
    # Parses through downloaded article
    # Pulls title and text from article
    if article.download_state == 2:
        article.parse()
        title = article.title
        text = article.text

        # Create output row
        new_title = pd.DataFrame(
            [[row["GlobalEventID"], row["SOURCEURL"], title, text,
              row["match30"], row["match50"], row["match70"]]],
            columns=colnames)
        count_success += 1

    else:
        # print(f'{row['SOURCEURL']} failed')
        new_title = pd.DataFrame(
            [[row["GlobalEventID"], row["SOURCEURL"], "NA", "NA",
              row["match30"], row["match50"], row["match70"]]],
            columns=colnames)
        count_fail += 1

    # Append new row to list
    df_list.append(new_title)

    if row_count % CHUNK_SIZE == 0:
        print("Another 1000 articles processed. Script running for",
              time.strftime("%H hr : %M min : %S sec",
                            (time.gmtime(time.time() - start_time))))

        # Concatenate data frames
        articles_df = pd.concat(df_list, ignore_index=True)

        # Create output file name
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        fn = FN_OUT.format(first_row, row_count, time_str)
        fn = os.path.join(DF_OUT, fn)
        print("Writing file to csv:\n {}".format(fn))

        # Save CSV
        articles_df.to_csv(fn, index=False)

        # Re-initialize list
        df_list = []

        # Set row counter (this will be the first row of the next file)
        first_row = row_count + 1

# Sends title and text of each article to a .csv
articles_df.to_csv(FN_OUT, index=False)
print("\nNumber of articles sucessfully downloaded:", count_success)
print("\nNumber of broken URLs:", count_fail)
print("Total time elapsed:\n",
      time.strftime("%H hr : %M min : %S sec",
                    (time.gmtime(time.time() - start_time))))