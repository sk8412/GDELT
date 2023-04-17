# code modified from: https://github.com/walfaelschung/GDELT_flow#readme

# Imports 
import pandas as pd
from pandas.errors import EmptyDataError
import zipfile
from zipfile import BadZipFile
import requests
from io import BytesIO
import time

# Pulls zip files from GDELT repository
# Filters by events data and date
master = pd.read_csv("http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt", sep=" ", header=None,  dtype={0: str}) # note: separator is blank space
master = master[master[2].str.contains('translation.export.CSV', na=False)] # column 2 contains the files. file names containing "export" refer to events data
master = master[master[2].str.contains('20220310', na=False)] # filters by date in zip filename

urls = list(master[2])

# Specify month, year, and countries of interest
MonthYeartoSort = ['202203']
fipscodes = ['UP']      

# Column headers for .csv
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
result_df2 = pd.DataFrame(columns=colnames)
protest_df = pd.DataFrame(columns=colnames)


finished_files =[]
file_errors = []

# Initializing counters and time
COUNT = 0
start_time = time.time()
TIMECOUNTER = 0

# Looping through each selected zip file
for a in urls:
    TIMECOUNTER = TIMECOUNTER +1

    # Prints updates on progress
    # Creates a backup of last 1000 files processed
    # Updates list of all files processed thus far
    if TIMECOUNTER == 1000:

        now_time = time.time()
        print('Another 1000 files processed. Script running for {0} seconds.'.format(now_time - start_time))
        print('Writing a backup copy to csv...')
        result_df.to_csv('results_backup.csv',index=False, sep='\t')
        protest_df.to_csv('results_all_protest.csv',index=False, sep='\t')
        TIMECOUNTER = 0

    # Prints updates on progress
    print('Handling file '+str(COUNT)+' of '+str(len(urls)))
    COUNT = COUNT+1
    obj = requests.get(a).content
    try:
        zf = zipfile.ZipFile(BytesIO(obj), 'r')
        filename = zf.namelist()
        
        # Opens each zip file and pulls any entries that meet criteria
        # Criteria: 14 = protest event; fipscodes = countries specified in line 22
        with zf.open(filename[0], 'r') as csvfile:
            try:
                df = pd.read_csv(csvfile, sep="\t", header=None, dtype={26: str,27:str,28:str})
                df.columns=colnames
                protestdf = df.loc[df.EventBaseCode.str.startswith('14', na=False)]  # EVENT-FILTER HERE
                protest_df = pd.concat([protest_df, protestdf])
                df_to_add = protestdf.loc[protestdf.ActionGeo_CountryCode.isin(fipscodes)] # COUNTRY-FILTER HERE
                
                # If df_to_add is empty, moves to next file
                if df_to_add.empty:
                    continue

                # Filters by date entered in line 21
                result_df = pd.concat([result_df, df_to_add])
                df_to_add2 = df_to_add.loc[df_to_add.MonthYear.isin(MonthYeartoSort)] # YEARMONTH-FILTER HERE
                result_df2 = pd.concat([result_df2, df_to_add2])
            except EmptyDataError:
                print('File was empty, moving on...')
                file_errors.append(a)
            result_df.to_csv('results_filtered.csv',index=False,sep='\t')   # Sends filtered entries to a .csv

    except BadZipFile:
        file_errors.append(a)
        print('Could not open zip file. Moving on...')
    finished_files.append(a)


# Checks for duplicate entries and runs basic descriptives
result_final = result_df.drop_duplicates(subset = ['GlobalEventID'])
print(result_final.groupby(['ActionGeo_CountryCode']).size())
print(result_final.Day.min())
print(result_final.Day.max())
print(len(result_final))