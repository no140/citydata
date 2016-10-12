#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import requests
import geocoder
import time

city = "NEW YORK"
if city == "NEW YORK":
	borough = 'MANHATTAN'
else: borough = city
parms={'$select': 'boro,housenumber,streetname,zip', 'boro': borough}
url_hpd = "https://data.cityofnewyork.us/resource/sc6a-36xc.json"
print('requesting HPD data...')
hpdr = requests.get(url_hpd, params=parms)
print('request complete, now reading json...')
hpdf=pd.read_json(hpdr.text, dtype={'zip': 'str'})
hpdf['address'] = hpdf.housenumber+' '+ hpdf.streetname + ', ' + city+', NEW YORK'# hpdf.boro + ', ' + hpdf.zip
hpdf.dropna()
#url_geo = 'https://maps.googleapis.com/maps/api/geocode/json'
longlist = []
latlist = []
pause = 5
for i,row in hpdf.iterrows():#_using.iterrows():
	#geops = {'sensor': 'false', 'address': row['address'], 'key': API_KEY_GEO}
	#geor = requests.get(url_geo, params=geops)
	#georesponse = geor.json()#['results']
	#print(row['address'])
	try:
		g = geocoder.osm(row['address'])
	except TypeError as e:
		print('TypeError:')
		print(e)
		print('lets pause for %d seconds..' % pause)
		time.sleep(pause)
		pause += 5
		pass
	except:
		print('problem in getting osm response')
		time.sleep(pause)
		pause += 1
		pass
	#print(g.osm)
	if g.osm:
		#hpdf['long'][i] = g.osm["x"]
		#hpdf['lat'][i] = g.osm["y"]
		longlist.append(g.osm["x"])
		latlist.append(g.osm["y"])
	else: 
		print('no osm')
		#hpdf['long'][i] = None
		#hpdf['lat'][i] = None
		longlist.append(None)
		latlist.append(None)
hpdf['long'] = longlist
hpdf['lat'] = latlist
hpdf.to_csv('hpd.csv')