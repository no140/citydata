#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

df = pd.read_csv('./input/311_Service_Requests_from_2010_to_Present_NY.csv', nrows=250000)

print(df.head())

first = df[['Created Date','Closed Date', 'Agency', 'Incident Zip']]
print(first.head())
first['Created Date'] = pd.to_datetime(first['Created Date'], infer_datetime_format=True, errors='coerce')
first['Closed Date'] = pd.to_datetime(first['Closed Date'], infer_datetime_format=True, errors='coerce')
print(first.head())
#first = first['Created Date'].apply(lambda x: [pd.Timestamp(ts) for ts in x])
#first = first['Closed Date'].apply(lambda x: [pd.Timestamp(ts) for ts in x])
first['time'] = np.where(first['Closed Date'].notnull(), (first['Closed Date'] - first['Created Date'])/ np.timedelta64(1, 'D'), pd.to_timedelta('90 days')/ np.timedelta64(1, 'D'))
print(first.tail())
#fix zips to 5 digits
for count,x in enumerate(first['Incident Zip']):
	if type(x) is str:
		if len(x)>5:
			first['Incident Zip'][count]=x[0:5]

#first['Incident Zip'].apply(str)
#first['Incident Zip'].apply(lambda x: x[0:5] if type(x) is str and len(x)>5 else x)
#fix negative time values
#first['time'].apply(lambda x: abs(x) if x<0 else x)
#remove agencies
agency_list = ['DEP', 'DOB', 'DOT', 'HPD', 'NYPD', 'DSNY', 'FDNY', 'DPR', 'DSNY']
first = first[first.Agency.isin(agency_list)]
first.ix[first.time<0,'time'] = pd.to_timedelta('90 days')/ np.timedelta64(1, 'D')
second = first.groupby(('Agency','Incident Zip'), as_index=False).mean()#.mean()#
print(second)
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(second['time'],second['Incident Zip'])
ax.set_title('average city response times by zip code')#meterName+'\nFrom '+startDate.strftime('%Y-%m-%d')+' to '+endDate.strftime('%Y-%m-%d'))
#ax.set_ylim((0,1))
ax.set_xlabel('zip codes')
ax.set_ylabel('average response times')
#fig.savefig('uptime_by_circuit_'+meterName+'.pdf')
'''
'''
for area, grp, time in second[['Incident Zip','Agency']:
	print('hello')
	plt.plot(grp['Agency'], label=area)
	#grp['D'] = pd.rolling_mean(grp['B'], window=5)
	plt.plot(grp['time'], label='response time in days'.format(k=area))
	plt.legend(loc='best')
'''
#colors=['g','r','m','k','y','c','orange','b',] #dep,dob,dot,dpr,dsny,fdny,hpd,nypd
colors=['b','r','c','g','m','y','orange','k']
#colors = cm.rainbow(np.linspace(0, 1, len(second.groupby('Agency').mean())))
count=0
for key,grp in second.sort_values(['Agency','Incident Zip']).groupby('Agency'):
	#print('plotting')
	print(grp)
	plt.scatter(grp['Incident Zip'], grp['time'], label=key, color=colors[count])#reverse x,y on scatter
	count+=1
axes = plt.gca()
axes.set_ylim([0,200])	#reverse x.y for scatter #use 255 for 20000 rows
axes.set_xlim([10000,10040])
#locs,labels = plt.xticks()
#axes.set_xticks(locs,labels)
#y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
#ax.yaxis.set_major_formatter(y_formatter)
axes.get_xaxis().get_major_formatter().set_useOffset(False)
#from matplotlib import rcParams
#rcParams['xtick.direction'] = 'out'
#rcParams['ytick.direction'] = 'out'
axes.tick_params(axis='y', direction='out')
axes.tick_params(axis='x', direction='out')
axes.set_xlabel('zip codes')
axes.set_ylabel('average response times (days)')
plt.legend(loc='upper right', fontsize='small')#lower right for scatter
plt.savefig('./output/scatter_ny.png')
plt.show()
plt.close()
