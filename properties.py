#!/usr/bin/env python3

import pandas as pd
print(pd.__version__)
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import geopy as gp

#csv = './input/Historic_Secured_Property_Tax_Rolls.csv'
csv = r"""dummy,date,loc,x
bar,20090101,a,1
bar,20090102,a,3
bar,20090103,a,5
bar,20090101,b,1
bar,20090102,b,3
bar,20090103,b,5"""

df = pd.read_csv('./input/Historic_Secured_Property_Tax_Rolls.csv')#, index_col=[0], use_cols=[0,3,4,5,7,8,9,11,14,20,22,35,37,38,40,43])
'''
print(df.head())
print(df.index)
print(df.columns)
print(df.loc[0:5,['Closed Roll Fiscal Year','Neighborhood Code','Neighborhood Code Definition','Block and Lot Number','Property Class Code','Property Class Code Definition','Year Property Built','Number of Bedrooms','Number of Rooms','Number of Units','Property Area in Square Feet','Lot Area','Current Sales Date','Closed Roll Assessed Improvement Value','Closed Roll Assessed Land Value','Zipcode of Parcel','Location']])
'''
#print(df.groupby('Property Class Code').head())
print(df.groupby('Property Class Code').size())
print('fraction: ',max(df.groupby('Property Class Code').size())/sum(df.groupby('Property Class Code').size()))

g = df.groupby('Block and Lot Number', as_index=False)
first = g.nth(0)#, dropna='any')	#gives first instance of BnL only
#print(first.size())	#see how it shortened
#print(first)

print(first['Closed Roll Assessed Improvement Value'].shape)

improv = first[first['Closed Roll Assessed Improvement Value']!=0]

print(improv[['Neighborhood Code','Closed Roll Assessed Improvement Value']].groupby('Neighborhood Code').mean()) #.filter(lambda x: len(x['Closed Roll Assessed Improvement Value'])!=0)
#print(df['Closed Roll Assessed Improvement Value'>0].groupby('Neighborhood Code').mean())

print('difference: ',max(improv[['Neighborhood Code','Closed Roll Assessed Improvement Value']].groupby('Neighborhood Code').mean()['Closed Roll Assessed Improvement Value']) - min(improv[['Neighborhood Code','Closed Roll Assessed Improvement Value']].groupby('Neighborhood Code').mean()['Closed Roll Assessed Improvement Value']))

locs = first[pd.notnull(first['Location'])]
locs = locs[['Neighborhood Code','Location']]
#locs['lat'], locs['long'] = zip(*locs['Location'].apply(lambda x: gp.Point(x).split(': ', 1)))
def geopt(x):
	return gp.Point(x).latitude, gp.Point(x).longitude
locs['lat'],locs['long'] = zip(*locs['Location'].map(geopt))
#locs['Location'] = locs.Location.apply(lambda x: pd.Series({'lat':gp.Point(x).latitude, 'long':gp.Point(x).longitude}))
#locs = locs[['Neighborhood Code','lat','long']]
#for item in locs.Location:
#	print(gp.Point(item).latitude, gp.Point(item).longitude)
#locs = locs[len(locs['Location'])==2]
#locs['lat'], locs['long'] = zip(*locs.Location)
#locs['lat'] = gp.Point(locs.Location).latitude
#locs['long'] = gp.Point(locs.Location).longitude
print(locs.shape)
print(locs.head())
#coords = locs[locs['Location']!=0]
#print(coords.shape)
stds = locs[['Neighborhood Code','lat','long']].groupby('Neighborhood Code').std()
#area of ellipse is pi*radius_x*radius_y
#minute of lat is generallistically ~111.03km, long~1.42km
stds['area'] = np.pi*stds.lat*stds.long*111.03*1.42
print(max(stds.area))

#bedrooms/unit ratio
beds = first[['Zipcode of Parcel','Number of Bedrooms', 'Number of Units']]
beds = beds[beds['Number of Bedrooms']!=0]# and 
beds = beds[beds['Number of Units']!=0]
bpu = beds.groupby('Zipcode of Parcel').mean()
print(bpu.head())
bpu['ratio'] = bpu['Number of Bedrooms']/bpu['Number of Units']
print('max ratio: ',max(bpu.ratio))

#median improvement
print(improv['Closed Roll Assessed Improvement Value'].head())
print('median: ',improv['Closed Roll Assessed Improvement Value'].median())

#built-up
build = first[['Zipcode of Parcel','Property Area in Square Feet','Lot Area']]
build = build[pd.notnull(build['Property Area in Square Feet'])]
build = build[pd.notnull(build['Lot Area'])]
build = build[build['Property Area in Square Feet']!=0]
build = build[build['Lot Area']!=0]
buildup = build.groupby('Zipcode of Parcel').sum()
print(buildup.head())
buildup['ratio'] = buildup['Property Area in Square Feet']/buildup['Lot Area']
print('largest ratio: ',max(buildup.ratio))

#avg units before/after 1950 build year - use earliest record
last = g.nth(-1)	#use last for earliest record
units = last[['Year Property Built','Number of Units']]
units = units[units['Year Property Built']<=2015]
units = units[units['Year Property Built']>=1700]
units = units[units['Number of Units']!=0]
units['thresh'] =  units['Year Property Built'].map(lambda x: True if x>=1950 else False)
#avg = units.groupby('Year Property Built').mean()
avg = units.groupby('thresh').mean()
#avg['thresh'] =  avg['Year Property Built'].map(lambda x: True if x>=1950 else False)#np.where(avg['Year Property Built']>=1950, True, False)
print(avg)

p = df[['Closed Roll Fiscal Year','Closed Roll Assessed Land Value']]
pgrowth = p.groupby('Closed Roll Fiscal Year').mean()
print(pgrowth)
growth_rate = np.exp(np.diff(np.log(pgrowth['Closed Roll Assessed Land Value']))) - 1
print(growth_rate)
print(np.average(growth_rate))
#print(pgrowth.iloc[:7])
#print(np.polyfit(pgrowth.iloc[:7]['Closed Roll Assessed Land Value'],growth_rate,1))

