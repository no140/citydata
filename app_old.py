#!/usr/bin/env python3

import pandas as pd
import numpy as np

from ediblepickle import checkpoint
import os
import urllib2


cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

@checkpoint(key=lambda args, kwargs: urllib2.quote(args[0]) + '_' + str(args[1]) + '.p', work_dir=cache_dir)
def get_data(city, rows, where):
    params = { #'format'        :'json', 
               '$limit':         rows, 
               'city' :          city,
                '$where' :         where}
    print 'making API request...'
    result = requests.get(url, params=params)
    print 'API request complete.'
    return result

city = 'NEW YORK'
calls = 500
agencies = ("DEP","DOB","DOT","HPD","NYPD","DSNY","FDNY","DPR")

agencyList = ','.join('"%s"' % x for x in agencies)

r = get_data(city,calls,'agency in('+agencyList+')')

df = pd.read_json(r.text, convert_dates=True)
print(df.head())
print(len(df))
print(df.columns.tolist())
#clean up database
collist = ['created_date', 'closed_date', 'agency', 'incident_zip']
newcols = ['created', 'closed', 'agency','zip']
df = df[collist]
#rename cols
df.rename(columns=dict(zip(collist, newcols)), inplace=True)
print(df.columns[0])

df['time'] = np.where(df.closed.notnull(), pd.to_datetime(df.closed, infer_datetime_format=True, errors='coerce') - pd.to_datetime(df.created, infer_datetime_format=True, errors='coerce'),90)
df.ix[df.time<0,'time'] = pd.to_timedelta('90 days')/ np.timedelta64(1, 'D')

print(df['time'].isnull().values.any())
print(df['time'].dtype)

grouped = df.groupby(['zip','agency'], as_index=False).agg({'time':avg_datetime, 'created': "count"})#'mean' didn't work on datetimes
print(grouped.head())

#using a fake input agency choice
ag_in = 'NYPD'
print(grouped[grouped.agency == ag_in].columns)#values


newgroups = grouped[grouped.agency == ag_in]
newgroups = newgroups[['zip','created']]
#for x in newgroups.values:
#    print x[0]

x = newgroups.values[0]#grouped[grouped.agency == ag_in].values[0]
y = newgroups.values[0]#grouped[grouped.agency == ag_in].values[2]


