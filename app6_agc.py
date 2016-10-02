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

from shapely.geometry import Point, shape
import json
import multiprocessing

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

@checkpoint(key=lambda args, kwargs: urllib.parse.quote(args[0]) + '_' + str(args[1]) + '.p', work_dir=cache_dir)#, refresh=True)
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
df = df.dropna()
print(df.head(5))

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


# load GeoJSON file containing sectors                                                                                                                                                                                                       
with open('../static/nynta.geojson', 'r') as f:
    js = json.load(f)
polygons = {}
for feature in js['features']:
	#print(feature['properties']['NTAName']) #if nynta, 'neighborhoods' for nychoods
	polygons[feature['properties']['NTAName']] = shape(feature['geometry'])
	
#print(polygons.items())
def which_neighb(latlong):
	'''Outputs NY neighborhood name given (lat,long).'''
	
	# construct point based on lat/long returned by the geocoder geocoder
	#point = Point(-87.65, 41.8)
	#print('about to convert to point...')
	point = Point(latlong[0],latlong[1])

	# check each polygon to see if it contains the point
	for neighb,polygon in polygons.items():
		#print(neighb)
		if polygon.contains(point):
			return neighb
	else: return None#'Not in New York!'

#print(df['lat'].dtype)
#print(df['long'].dtype)
#print(df['lat'][10],df['long'][10])
#print(which_neighb(df['long'][10],df['lat'][10]))
pool = multiprocessing.Pool()
neigh = pool.map(which_neighb, df[['long','lat']].apply(tuple, axis=1))
pool.close()
pool.join()
#print(neigh)
f.close()

df['hood'] = neigh
#df['hood'] = df.apply(which_neighb(df['long'],df['lat']), axis = 1)#np.vectorize(which_neighb(df['lat'],df['long']))# 
#df = df[df['hood' != 'Not in New York!']]
df = df.dropna()
print(df.head())

#use complaints instead of agency below for more granularity
grouped = df[['created','agency','hood','lat','long']].groupby(['hood','agency'], as_index=False).agg({'lat': 'mean', 'long':'mean','created': 'count'})#{'created': {'created': "count"}, 'latitude':{'lat': 'mean'}})#,'longitude':{'long', 'mean'}})
print(grouped.head(30))

#print(len(grouped[grouped.agency =="NYPD"]))
#print(len(grouped[grouped.agency =="DSNY"]))
agencygrp = df[['created','agency']].groupby(['agency'], as_index=False).count()
print(agencygrp)
ttlcalls = sum(agencygrp.created)
print(ttlcalls)
hoodgrp = df[['created','hood']].groupby(['hood'], as_index=False).count()
print(hoodgrp)
#print(hoodgrp.ix[hoodgrp.hood == "West Village", 'created'])
df0 = grouped.merge(agencygrp[['agency','created']], on='agency')
df0.rename(columns={'created_x':'created'}, inplace=True)
df0.rename(columns={'created_y':'agencyttl'}, inplace=True)
#print(df0)
df1 = df0.merge(hoodgrp[['hood','created']], on='hood')
df1.rename(columns={'created_x':'created'}, inplace=True)
df1.rename(columns={'created_y':'hoodttl'}, inplace=True)
print(df1)
#df2 = df1.assign(ratio = lambda x: ((x.created/x.hoodttl) / (x.agencyttl/ttlcalls)))
df2 = df1.assign(ratio = lambda x: ((x.created) / (x.agencyttl/ttlcalls)))
#print(df2.nlargest(3, columns='ratio'))
df3 = df2[['hood','agency','ratio','created','lat','long']]#
#df4 = df3['ratio'].groupby(level=0, group_keys=True)
df4 = df3.sort(['hood','ratio'],ascending=False).groupby('hood').head(3)
print(df4)
#print(df3.groupby('hood').sort_values(by='ratio', axis=0, ascending=False, inplace=True).head(3))

n_colors = len(agencies)
#colors = plt.cm.Spectral(np.linspace(0, 1, n_colors))
colors=['b','r','c','orange','k','m','y','g', 'grey','brown']	#"DEP","DOB","DOT","HPD","NYPD","DSNY","FDNY","DPR", "DOHMH","DHS" #dep,dob,dot,dpr,dsny,fdny,hpd,nypd
#print(colors)
plt.figure(num=None, figsize=(10, 16), dpi=80, facecolor='w', edgecolor='k')
#for i,col in zip(range(numcols),colors):
#	print(i, col)
X = df4.as_matrix(columns=df4.columns[1:])
print(X)
rmax = max(X[:,1])
df4['ratio'] = df4['ratio'].apply(lambda x: x/rmax)
xf = df4.as_matrix()
#ratmax = max(xf[:,1])

print(xf)

#with open('nynta.geojson') as data_file:    
#    data = json.load(data_file)
import pygeoj
nf = pygeoj.load("../static/nynta.geojson")
for i,feature in enumerate(nf):
	#print(len(nf)) # the number of features
	#print(nf.bbox) # the bounding box region of the entire file
	#print(nf.crs) # the coordinate reference system
	#print(nf.all_attributes) # retrieves the combined set of all feature attributes
	#print(nf.common_attributes) # retrieves only those field attributes that are common to all features
	#print(feature.properties['NTAName'])#.NTAName) #feature.geometry.type
	print(feature.properties['BoroName']) #feature.geometry.coordinates
	if feature.properties['BoroName'] == "Manhattan":
		#find neighborhood
		count = 1
		notFound = 1
		for row in xf:
			#print(row)
			if row[0] == feature.properties['NTAName'] and notFound:
				#update property
				key = "top"+str(count)
				key2 = "ag"+str(count)
				key3 = "count"+str(count)
				print('updating property...with '+key)
				feature.properties[key] = row[2] #{key: row[2]}#, "top2":(row+1][2], "top3": df4[row+2][2]}
				feature.properties[key2] = row[1]
				feature.properties[key3] = row[3]
				count += 1
				if count > 3:
					count = 1
					notFound = 0
					#break
	else: 
		print('deleting feature...')
		feature.properties["ag1"] = " "
		del nf[i]
nf.save('nynta_copy.geojson')

#print(agencyList)
for row in range(len(X)):#df4)):
	#print(agencyList.index(X[row][0]))
	ag = X[row][0]
	if ag == "DEP":#colors[agencyList.index(X[row][0])])
		col = 'c'
	elif ag == "DPR":
		col = 'g'
	elif ag == "DOT":
		col = 'b'
	elif ag == "DSNY":
		col = 'm'
	elif ag == "FDNY":
		col = 'y'
	elif ag == "HPD":
		col = 'orange'
	elif ag == "DOB":
		col = 'r'
	else: col = 'k'
	alpha = X[row][1]
	if alpha > 1.0: alpha =1
	plt.plot(X[row][3],X[row][2],'o', mfc=col, markersize=10, alpha=alpha, label='_nolegend_' if alpha<0.5 else ag)
	#plt.plot(df4.iloc[row].long,df4.iloc[row].lat,'o',markerfacecolor=colors[agencyList.index(df4.iloc[row].agency)])#, alpha=df4.ratio, markeredgecolor='k', markersize=10)
plt.title('Manhattan neighborhoods top 3 types of 311 calls')
plt.axes().set_aspect('equal', 'datalim')
plt.axes().set_ylim([40.70,40.88])
plt.axes().set_xlim([-74.03,-73.90])
plt.legend(numpoints=1,scatterpoints=1, scatteryoffsets=[0.0], markerscale=1.0, frameon=False, loc='lower right', fontsize='small')
plt.savefig('../output/top3_complainttypes_'+str(calls)+'.png')
plt.show()
plt.close()


#----------------------------------------------------------------------------------
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
'''
factor = 0.00001
agencydict = {"DEP": factor*2,"DOB": factor*5,"DOT": factor*3,"HPD": factor*6,"NYPD": factor*8,"DSNY": factor*4,"FDNY": factor*7,"DPR": factor*1, "DOHMH": factor*9, "DHS": factor*10}
df['agnums'] = [agencydict[a] for a in df.agency.values]
'''

'''
a = CountVectorizer()
aX = a.fit_transform(df['agency'])
#make differences tiny by multiplying by factor:
aX = aX*factor
#complaints = cX.transform(df['complaint'])
'''
'''
vectorlist=[]
for x in df['complaint'].values:
	vectorlist.append({x: 1})#for x in df['complaint'].values})
	#vectorlist.append({j: 1 for j in x})
	#print(x)	
#print(vectorlist)
print(len(vectorlist))
v = DictVectorizer()
cX = v.fit_transform(vectorlist)
#make differences tiny by multiplying by factor:
cX = cX*factor
print(cX.shape)
'''
'''
testcols = ['lat', 'long']#,'agnums']# 
tests = df.as_matrix(columns=testcols)
#for i in range(len(tests)):
#	tests[i][0] -= 40
#	tests[i][1] +=74
#[tests[0][x] = (x-40) for x in tests[0][range(len(tests))]]
#tests[1][x] = [(x+74) for tests[1][x] in range(len(tests))]
'''
'''
print(X.shape)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.01 * (1 - .01)))
sel.fit_transform(X)
print(X.shape)
#find rows selected
selmask = sel.get_support()
'''
'''
c = CountVectorizer()
cX = c.fit_transform(df['complaint'])	#df['agency'])
#complaints = cX.transform(df['complaint'])
#make differences tiny by multiplying by factor:
#factor = 0.00000005
#aX = aX*factor
'''
'''
import scipy.sparse as sp
X = sp.hstack((tests,aX), format='csr')
X = X.todense()
print(X.shape)
print(X[0:5][0:5])
'''
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
pca.fit(cX.T.toarray())
#X = tests
print(pca.explained_variance_ratio_) 
print(pca.components_.shape)
print(len(pca.components_))
#from sklearn.preprocessing import normalize
#pca_componenets_=normalize(pca.components_, norm='l1',copy=False)
col_max = pca.components_.max()
col_min = pca.components_.min()
print(col_max, col_min)
#normalize
def norm(x):
    return ((x-col_min) / (col_max - col_min))
normalize = np.vectorize(norm)
pca.components_ = normalize(pca.components_)
print(tests.shape)
X = np.column_stack([tests, pca.components_.T])
print(X)
'''
'''
numcols = len(pca.components_)
colors = plt.cm.Spectral(np.linspace(0, 1, numcols))
#print(colors)
plt.figure(num=None, figsize=(10, 16), dpi=80, facecolor='w', edgecolor='k')
#for i,col in zip(range(numcols),colors):
#	print(i, col)
for row in range(len(X)):
	for col in range(2,numcols):
		if X[row][col] > 0:
			#print(X[row][col])
			plt.plot(X[row,1],X[row,0],'o',markerfacecolor=colors[col], 					alpha=X[row][col], markeredgecolor='k', markersize=8)
'''
'''
eps = 0.007#0.008
samples = 10#int(round(calls/500))	#~50-100
db = DBSCAN(eps=eps, min_samples=samples, algorithm='ball_tree', leaf_size=10).fit(X)
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

groups = []#[[[0 for n in range(n_clusters-1)]for row in range(len(X))] for x in range(len(X[0]))]
#
plt.figure(num=None, figsize=(10, 16), dpi=80, facecolor='w', edgecolor='k')
for k, col in zip(unique_labels, colors):
	if k == -1:
		# Black used for noise. > or gray with transparency
		col = '0.6'#'k'
		trans = 0.05
	else: trans = 1.0

	class_member_mask = (labels == k)
	#print(len(class_member_mask & core_samples_mask))
	group = df[class_member_mask & core_samples_mask][['lat','long','agency','complaint']]
	
	grouped = group.groupby(['complaint'], as_index=False)#['created'].count()
	print(grouped.head(5))
	#grouped.sort_values(by='count', axis=0, ascending=False, inplace=True)

	xy = X[class_member_mask & core_samples_mask]
	plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col, alpha=trans, label=k,
	     markeredgecolor='k', markersize=14)

	#if k == 0:
	#	groups = group
	#	print(group.shape)
	#if k != -1 and k!= 0:
	#	print(group.shape)
		#np.stack((groups,group))

	xy = X[class_member_mask & ~core_samples_mask]
	plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col, alpha=trans,
	     markeredgecolor='k', markersize=6)

#group by cluster, going back to data within df
#print(groups.shape)
#grouping = df[
'''
'''
#plt.title('Manhattan with %d estimated clusters and %.3f eps\n from %s to %s' % (n_clusters, eps, startdate, enddate))
plt.axes().set_aspect('equal', 'datalim')
plt.axes().set_ylim([40.70,40.88])#[40.61, 40.91])([.7,.88])
plt.axes().set_xlim([-74.03,-73.90])#[-74.06,-73.77])([-.03,.1])
#plt.legend(scatterpoints=1, scatteryoffsets=[0.5], markerscale=30/n_clusters, frameon=False, loc='lower right', fontsize='small')
#plt.savefig('../output/dbSCAN_types_'+str(calls)+'_clusters_'+str(n_clusters)+'.png')#+'_'+str(eps)
plt.show()
plt.close()
'''
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

