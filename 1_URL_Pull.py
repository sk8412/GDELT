# code modified from: https://github.com/walfaelschung/GDELT_flow#readme

# Imports 
import pandas as pd
from pandas.errors import EmptyDataError
import zipfile
from zipfile import BadZipFile
import requests
from io import BytesIO
import time
import os
from datetime import datetime

# Specify year of interest
# Can be edited to include month, ex: 202203
YeartoSortZip = '2022'
YeartoSortEntry = '2022'

# Rows per file
CHUNK_SIZE = 1000

# Report frequency
PRINT_FREQ = 10

# Initializing counter
ENTRY_COUNT = 1

# Output file name
FN_OUT = '/Protest_Events_All.csv'

# Output backup file names
FN_OUT_BU = 'Protest_Events_{:09d}_{:09d}_{}.csv'

# Output data folder
DF_OUT = '/home/skyler/gdelt_protest_events'

# Pulls zip files from GDELT repository
# Filters by events data and date in zip filename
master = pd.read_csv('http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt', sep=' ', header=None,  dtype={0: str}) # note: separator is blank space
master = master[master[2].str.contains('translation.export.CSV', na=False)] # column 2 contains the files. file names containing 'export' refer to events data
master = master[master[2].str.contains(YeartoSortZip, na=False)] # filters by date in zip filename

urls = list(master[2])

# Column headers for CSV
colnames = ['GlobalEventID', 'Day','MonthYear','Year','FractionDate',
            'Actor1Code','Actor1Name','Actor1CountryCode','Actor1KnownGroupCode',
            'Actor1EthnicCode','Actor1Religion1Code','Actor1Religion2Code',
            'Actor1Type1Code','Actor1Type2Code','Actor1Type3Code',
            'Actor2Code','Actor2Name','Actor2CountryCode','Actor2KnownGroupCode',
            'Actor2EthnicCode','Actor2Religion1Code','Actor2Religion2Code',
            'Actor2Type1Code','Actor2Type2Code','Actor2Type3Code',
            'IsRootEvent','EventCode','EventBaseCode','EventRootCode',
            'QuadClass','GoldsteinScale','NumMentions','NumSources',
            'NumArticles','AvgTone',
            'Actor1Geo_Type','Actor1Geo_Fullname',
            'Actor1Geo_CountryCode','Actor1Geo_ADM1Code','Actor1Geo_ADM2Code',
            'Actor1Geo_Lat','Actor1Geo_Long','Actor1Geo_FeatureID',
            'Actor2Geo_Type','Actor2Geo_Fullname',
            'Actor2Geo_CountryCode','Actor2Geo_ADM1Code','Actor2Geo_ADM2Code',
            'Actor2Geo_Lat','Actor2Geo_Long','Actor2Geo_FeatureID',
            'ActionGeo_Type','ActionGeo_Fullname',
            'ActionGeo_CountryCode','ActionGeo_ADM1Code','ActionGeo_ADM2Code',
            'ActionGeo_Lat','ActionGeo_Long','ActionGeo_FeatureID',
            'DATEADDED','SOURCEURL'
            ]

# Initializing the dataframes
result_df = pd.DataFrame(columns=colnames)
filter_date_df = pd.DataFrame(columns=colnames)

# Initializing the lists
finished_files =[]
file_errors = []
df_list = []

# Initializing time
start_time = time.time()

# Looping through each selected zip file
for a in urls:
    obj = requests.get(a).content

    # Prints progress going through zip files
    if ENTRY_COUNT % PRINT_FREQ == 0:
        print('Processing ' + 
              str(ENTRY_COUNT) + ' of ' + str(len(urls)))
        
    try:
        zf = zipfile.ZipFile(BytesIO(obj), 'r')
        filename = zf.namelist()
        
        # Opens each zip file and pulls any entries that meet criteria
        # Criteria: 14 = protest event
        with zf.open(filename[0], 'r') as csvfile:
            try:
                df = pd.read_csv(csvfile, sep='\t', header=None, dtype={26: str,27:str,28:str})
                df.columns=colnames
                df['MonthYear'] = df['MonthYear'].astype('string')
                protest_df = df.loc[df.EventBaseCode.str.startswith('14', na=False)]  # EVENT-FILTER HERE
                filtered_by_date = protest_df.loc[protest_df.Year.str.fullmatch(YeartoSortEntry, na=False)] # YEAR-FILTER HERE
                result_df = pd.concat([result_df, filtered_by_date])

            except EmptyDataError:
                print('File was empty, moving on...')
                file_errors.append(a)
            ENTRY_COUNT += 1

    except BadZipFile:
        file_errors.append(a)
        print('Could not open zip file. Moving on...')
    finished_files.append(a)

    # Append new row to list
    df_list.append(result_df)

    # Saves backup files
    if ENTRY_COUNT % CHUNK_SIZE == 0:
        print('Another 1000 entries processed. Script running for',
              time.strftime('%H hr : %M min : %S sec',
                            (time.gmtime(time.time() - start_time))))

        # Concatenate data frames
        filtered_df = pd.concat(df_list, ignore_index=True)

        # Create output file name
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        fn = FN_OUT_BU.format(ENTRY_COUNT, len(urls), time_str)
        fn = os.path.join(DF_OUT, fn)
        print('Writing file to csv:\n {}'.format(fn))

        # Save CSV
        result_df.to_csv(fn, index=False)

        # Re-initialize list
        df_list = []

        # Set row counter (this will be the first row of the next file)
        first_row = ENTRY_COUNT + 1

# Checks for duplicate entries
result_final = result_df.drop_duplicates(subset = ['GlobalEventID'])

# Sends entries from GDELT database to a CSV
result_final.to_csv(DF_OUT + FN_OUT, index=False)
print('\nNumber of folders opened:', ENTRY_COUNT)
print('Total time elapsed:\n',
    time.strftime('%H hr : %M min : %S sec',
                    (time.gmtime(time.time() - start_time))))


# Basic descriptives
print(result_final.groupby(['ActionGeo_CountryCode']).size())
print('Starting date:', result_final.Day.min())
print('Ending date:', result_final.Day.max())
print('Number of entries compiled:', len(result_final))