#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

numrows=250000	#use 10k for testing
year = str(2016)
input_file = '../input/311_manhattan_'+year+'.csv'
df = pd.read_csv(input_file)	#allrows, nrows=numrows)#250000

print(df.head())

first = df[['Created Date','Closed Date', 'Agency','Latitude','Longitude']]
first.columns = ['Created Date','Closed Date','Agency','lat','long']
#first.rename(columns={'Incident Zip','zip'}, inplace=True)
first[['lat','long']]=first[['lat','long']].apply(pd.to_numeric)
print(first.head())
first['Created Date'] = pd.to_datetime(first['Created Date'], infer_datetime_format=True, errors='coerce')
first['Closed Date'] = pd.to_datetime(first['Closed Date'], infer_datetime_format=True, errors='coerce')
print(first.head())
#first = first['Created Date'].apply(lambda x: [pd.Timestamp(ts) for ts in x])
#first = first['Closed Date'].apply(lambda x: [pd.Timestamp(ts) for ts in x])
first['time'] = np.where(first['Closed Date'].notnull(), (first['Closed Date'] - first['Created Date'])/ np.timedelta64(1, 'D'), pd.to_timedelta('90 days')/ np.timedelta64(1, 'D'))
print(first.tail())

#remove agencies
agency_list = ['DEP', 'DOB', 'DOT', 'HPD', 'NYPD', 'DSNY', 'FDNY', 'DPR', 'DSNY']
#use agency:
#agency='NYPD'
first = first[first.Agency.isin(agency_list)]

#fix negative time values
first.ix[first.time<0,'time'] = pd.to_timedelta('90 days')/ np.timedelta64(1, 'D')
#group by
#second = first.groupby(('Agency'), as_index=False)#.mean()#
#print(second)

#pd.options.display.mpl_style = 'default' #Better Styling  
#new_style = {'grid': False} #Remove grid  
#matplotlib.rc('axes', **new_style)  
from matplotlib import rcParams  
rcParams['figure.figsize'] = (10, 17) #Size of figure  
rcParams['figure.dpi'] = 250


axes=plt.gca()
#fig, ax = plt.subplots(1, 1)
#axes.set_axis_bgcolor('black') #Background Color
#print('df type: ',second.dtype)'

colors=['b','r','c','g','m','y','orange','k']	#dep,dob,dot,dpr,dsny,fdny,hpd,nypd
oldkey='DEP'
count=0
for name,group in first.groupby('Agency'):#.iterrows():
	#if name=='DEP':
	if name != oldkey:
		count+=1
		oldkey=name
	plt.scatter(group.long, group.lat,color=colors[count],label=name, s=.02,alpha=.6)

#plt.scatter(first.long, first.lat, color='k', s=.02, alpha=.6)
#axes.set_title('311 complaints - '+year+' - '+agency)

axes.set_ylim([40.70,40.88])#[40.61, 40.91])
axes.set_xlim([-74.03,-73.90])#[-74.06,-73.77])
axes.tick_params(axis='y', direction='out')
axes.tick_params(axis='x', direction='out')
axes.set_xlabel('longitude')
axes.set_ylabel('latitude')
plt.legend(scatterpoints=1, scatteryoffsets=[0.5],markerscale=15.0,frameon=False,loc='lower right')#, fontsize='small')
plt.savefig('../output/scatter_ny_map_'+year+'.png')
plt.close()

'''
colors=['b','r','c','g','m','y','orange','k']	#dep,dob,dot,dpr,dsny,fdny,hpd,nypd
#colors = cm.rainbow(np.linspace(0, 1, len(second.groupby('Agency').mean())))

count=0
oldkey = 'DEP'	#must be starting agency...
for name,group in second.groupby('Agency'):#iterrows():#.itertuples():#(index=False):#.sort_values(['Agency','Incident Zip']):#.groupby('Agency'):
	#print('plotting')
	#print(index[0],index[1], time)#grp)#'key: ',key,'z: ',z,'t: ',t)#grp)
	print(name)#, group['Incident Zip'],group.time)
	#plt.scatter(index[1], time, label=index[0], color=colors[count])#reverse x,y on scatter
	
	#if key != oldkey:
	#print('agency: ',index[0])#[enum])
	#print('count ',count)
	if name != oldkey:#index[0][enum]
		count+=1
		print('updating count')
		oldkey = name#index[0]	#[enum]#key
	plt.plot(group['Incident Zip'],group.time, marker='o',linestyle='',label=name,color=colors[count])

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
plt.legend(numpoints=1, loc='upper right', fontsize='small')#lower right for scatter
plt.savefig('./output/scatter_ny_'+str(round(numrows/1000))+'k.png')
plt.show()
plt.close()
'''
