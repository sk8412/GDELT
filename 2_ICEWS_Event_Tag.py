# -*- coding: utf-8 -*-

import math
import time
import pandas as pd
from io import BytesIO


# Read raw bytes from file
def read_bytes(fn):
    with open(fn, 'rb') as fh:
        buf = fh.read()
    return buf


# Spherical distance between lon/lat points
def geo_dist(x1, y1, x2, y2, radius=6371):
    lon1, lat1, lon2, lat2 = map(math.radians, [x1, y1, x2, y2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2)**2 +
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2)
    c = 2 * math.asin(math.sqrt(a))
    d = radius * c
    return d


# Main data folder
# DF_MAIN = 'C:\\Users\\2bowb\\Documents\\Thesis\\'
# DF_MAIN = 'G:\\My Drive\\Theses\\Kyler\\'

# GDELT file name
FN_GDELT = 'results_filtered.csv'
# FN_GDELT = 'thesis-gdelt-false-positives-master\\results_all_protest.csv'

# ICEWS file name
FN_ICEWS = 'events.2022.20230106.tab'
# FN_ICEWS = 'events.2022.20230106.tab'

# Output file name
FN_OUT = 'results_tagged_v1.csv'

# Distance thresholds for matching events
DIST_RADIUS = [30, 50, 70]

# Frequency of status updates
PRINT_FREQ = 100

# Start timer
start_time = time.time()

# Read ICEWS Events
print('\nReading data file:\n  {}'.format(FN_ICEWS))
idat = read_bytes(FN_ICEWS)

# Read GDELT protests
print('\nReading data file:\n  {}\n'.format(FN_GDELT))
gdat = read_bytes(FN_GDELT)

# Filter ICEWS
idf = pd.read_csv(BytesIO(idat), sep='\t')
cond = idf['CAMEO Code'].astype(str).str.startswith('14')
idf = idf[cond].dropna(subset=['City', 'Longitude', 'Latitude'])
idf = idf.drop_duplicates(subset=['Event Date', 'Longitude', 'Latitude'])

# Create ICEWS grid w/ integer lat/lon
idict = {}

idf['grid_key'] = (
    idf['Event Date'] + '__' +
    idf['Longitude'].apply(int).apply(str) + '__' +
    idf['Latitude'].apply(int).apply(str)
    )

for index, row in idf.iterrows():
    key = row['grid_key']
    if key not in idict:
        idict[key] = []
    idict[key].append(row[['Longitude', 'Latitude']].to_dict())

# Filter GDELT
gdf = pd.read_csv(BytesIO(gdat), sep='\t')
cond = gdf['ActionGeo_FeatureID'].astype(str).apply(len) > 3
gdf = gdf[cond].dropna(
    subset=['ActionGeo_FeatureID', 'ActionGeo_Long', 'ActionGeo_Lat'])

# Create GDELT query columns
gdf['Day'] = gdf['Day'].astype(str)
gdf['date'] = (
    gdf['Day'].str[0:4] + '-' +
    gdf['Day'].str[4:6] + '-' +
    gdf['Day'].str[6:8]
    )
gdf['lon'] = gdf['ActionGeo_Long'].apply(int)
gdf['lat'] = gdf['ActionGeo_Lat'].apply(int)

# Create GDELT tag columns
for dr in DIST_RADIUS:
    gdf['match' + str(dr)] = pd.Series(dtype='Int64')

# Loop over GDELT records
count = 0
for index, row in gdf.iterrows():
    count += 1
    if count % PRINT_FREQ == 0:
        print("Tagging row {} of {}".format(count, len(gdf)))
    lon0 = row['lon']
    lat0 = row['lat']
    matches = {}

    # Loop over neighboring grid cells
    for x_offset in (-1, 0, 1):
        lon = lon0 + x_offset
        for y_offset in (-1, 0, 1):
            lat = lat0 + y_offset

            # Create query key with integer lat/lon
            gkey = row['date'] + '__' + str(lon) + '__' + str(lat)
            if gkey not in idict:
                continue

            # Get ICEWS events for this grid cell
            events = idict[gkey]

            # Loop over ICEWS events
            for event in events:

                # Calculate distance(km) between GDELT record and ICEWS event
                dist = geo_dist(
                    row['ActionGeo_Long'], row['ActionGeo_Lat'],
                    event['Longitude'], event['Latitude'])

                # Loop over distance thresholds
                for dr in DIST_RADIUS:

                    # Record matches at each threshold
                    if dist <= dr:
                        matches['match' + str(dr)] = 1

    # Record results for this GDELT row
    for dr in DIST_RADIUS:
        mkey = 'match' + str(dr)
        if mkey in matches:
            gdf.at[index, mkey] = 1
        else:
            gdf.at[index, mkey] = 0

# Write results to CSV
out_fn = FN_OUT
print("\nWriting results to: {}".format(out_fn))
gdf.to_csv(out_fn, index=False)

print('\nTime elapsed:')
print(time.strftime(
    "%H hr : %M min : %S sec",
    time.gmtime(time.time()-start_time)))
