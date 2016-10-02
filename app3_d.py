#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy import cluster
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn import datasets, linear_model, utils, preprocessing


from ediblepickle import checkpoint
import os
import urllib#2
import requests

url = 'https://data.cityofnewyork.us/resource/fhrw-4uyv.json'
url3 = 'https://data.cityofnewyork.us/resource/fhrw-4uyv.json?city=NEW YORK&$limit=3000&$where=agency in("DEP","DOB","DOT","HPD","NYPD","DSNY","FDNY","DPR")'
url_2010 = 'https://data.cityofnewyork.us/resource/mwbr-9zz9.json'
url_2011 = 'https://data.cityofnewyork.us/resource/knik-dax9.json'
url_2012 = 'https://data.cityofnewyork.us/resource/6nxj-n6t5.json'
url_2013 = 'https://data.cityofnewyork.us/resource/e8jc-rs3b.json'
url_2014 = 'https://data.cityofnewyork.us/resource/f364-y3pv.json'
url_2015 = 'https://data.cityofnewyork.us/resource/hemm-82xw.json'
url_2009 = 'https://data.cityofnewyork.us/resource/76rq-desm.json'
url_2008 = 'https://data.cityofnewyork.us/resource/ttef-akmb.json'
url_2007 = 'https://data.cityofnewyork.us/resource/bjsb-smxa.json'
url_2006 = 'https://data.cityofnewyork.us/resource/txvy-sgqz.json'
url_2005 = 'https://data.cityofnewyork.us/resource/xk2u-49gx.json'
url_2004 = 'https://data.cityofnewyork.us/resource/fred-eu2a.json'

cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

@checkpoint(key=lambda args, kwargs: urllib.parse.quote(args[0]) + '_' + str(args[1]) + '.p', work_dir=cache_dir)
def get_data(city, rows, where):
    params = { #'format'        :'json', 
               '$limit':         rows, 
               'city' :          city,
                '$where' :         where}
    print('making API request...')
    result = requests.get(url, params=params)
    print('API request complete.')
    return result

city = 'NEW YORK'
calls = 50000#100000
agencies = ("DEP","DOB","DOT","HPD","NYPD","DSNY","FDNY","DPR", "DOHMH","DHS")

agencyList = ','.join('"%s"' % x for x in agencies)

r = get_data(city,calls,'agency in('+agencyList+')')

df = pd.read_json(r.text, convert_dates=True)
print(df.head())
print(len(df))
print(df.columns.tolist())
#clean up database
collist = ['created_date', 'closed_date','agency', 'incident_zip','complaint_type', 'latitude', 'longitude']
newcols = ['created', 'closed', 'agency','zip', 'complaint', 'lat', 'long']
df = df[collist]
#rename cols
df.rename(columns=dict(zip(collist, newcols)), inplace=True)
#print(df.columns[0])
'''
model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))])
model.fit(X, y)
model.predict(X_pred)
'''
'''
testcols = ['agency', 'complaint', 'lat', 'long']
tests = df.as_matrix(columns=testcols)
print(tests[0:5,0:5])
#plot variance for each value for 'k' between 1,10
initial = [cluster.vq.kmeans(tests,i) for i in range(10,30)]
plt.plot([var for (cent,var) in initial])
plt.show()
'''

agencydict = {"DEP": 2,"DOB": 5,"DOT": 3,"HPD": 6,"NYPD": 8,"DSNY": 4,"FDNY": 7,"DPR": 1, "DOHMH": 9, "DHS": 10}
df['agnums'] = [agencydict[a] for a in df.agency.values]

df = df.dropna()
print(df.head(5))
'''
c = CountVectorizer()
cX = c.fit_transform(df['complaint'])
#complaints = cX.transform(df['complaint'])
'''
import datetime
startdate = df.created.iloc[0]
#print(startdate.year, startdate.month, startdate.day)
startdate = datetime.datetime.strptime(startdate, '%Y-%m-%dT%H:%M:%S.%f')
print('%d-%d-%d' % (startdate.month, startdate.day, startdate.year))
startdate = str(startdate.month)+'/'+str(startdate.day)+'/'+str(startdate.year)
enddate = df['created'].iloc[-1]
enddate = datetime.datetime.strptime(enddate, '%Y-%m-%dT%H:%M:%S.%f')
enddate = str(enddate.month)+'/'+str(enddate.day)+'/'+str(enddate.year)
print(enddate)
print(max(df.created))
print(min(df.created))

vectorlist=[]
for x in df['complaint'].values:
	vectorlist.append({x: 1})#for x in df['complaint'].values})
	#vectorlist.append({j: 1 for j in x})
	#print(x)	
#print(vectorlist)
print(len(vectorlist))
v = DictVectorizer()
cX = v.fit_transform(vectorlist)
print(cX.shape)

testcols = ['agnums', 'lat', 'long']
tests = df.as_matrix(columns=testcols)

import scipy.sparse as sp
X = sp.hstack((tests,cX), format='csr')
X = X.todense()
print(X.shape)
print(X[0:5][0:5])

eps = 0.008
samples = int(round(calls/500))	#~50-100
db = DBSCAN(eps=eps, min_samples=samples).fit(X)
#core_samples = db.core_sample_indices_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print('# of clusters: '+str(n_clusters))
# Black removed and is used for noise instead.
unique_labels = set(labels)
print(unique_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

plt.figure(num=None, figsize=(10, 16), dpi=80, facecolor='w', edgecolor='k')
for k, col in zip(unique_labels, colors):
	if k == -1:
		# Black used for noise. > or gray with transparency
		col = '0.6'#'k'
		trans = 0.05
	else: trans = 1.0

	class_member_mask = (labels == k)

	xy = X[class_member_mask & core_samples_mask]
	plt.plot(xy[:, 2], xy[:, 1], 'o', markerfacecolor=col, alpha=trans,
	     markeredgecolor='k', markersize=14)

	xy = X[class_member_mask & ~core_samples_mask]
	plt.plot(xy[:, 2], xy[:, 1], 'o', markerfacecolor=col, alpha=trans,
	     markeredgecolor='k', markersize=6)

plt.title('Manhattan with %d estimated clusters and %.3f eps\n from %s to %s' % (n_clusters, eps, startdate, enddate))
plt.axes().set_aspect('equal', 'datalim')
plt.axes().set_ylim([40.70,40.88])#[40.61, 40.91])
plt.axes().set_xlim([-74.03,-73.90])#[-74.06,-73.77])
plt.savefig('../output/dbSCAN_types_'+str(calls)+'_clusters_'+str(n_clusters)+'.png')#+'_'+str(eps)
plt.show()
plt.close()
'''
df['time'] = np.where(df.closed.notnull(), pd.to_datetime(df.closed, infer_datetime_format=True, errors='coerce') - pd.to_datetime(df.created, infer_datetime_format=True, errors='coerce'),90)
df.ix[df.time<0,'time'] = pd.to_timedelta('90 days')/ np.timedelta64(1, 'D')

print(df['time'].isnull().values.any())
print(df['time'].dtype)
'''
'''
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
y = newgroups.values[2]#grouped[grouped.agency == ag_in].values[2]
'''

