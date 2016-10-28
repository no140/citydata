#!/usr/bin/env python3
from os.path import dirname, join
from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from collections import Counter
from vardata import *

from shapely.geometry import Point, shape
import json
import multiprocessing
import gevent

from ediblepickle import checkpoint
import os
import urllib	#for python3, remove '2' and add .parse before quote
import datetime
from dateutil.relativedelta import relativedelta
import requests
from bokeh.plotting import figure, output_file, show
from bokeh.charts import TimeSeries, Scatter, defaults
from bokeh.layouts import gridplot
from bokeh.embed import components
from bokeh.palettes import Spectral6
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider
from bokeh.io import curdoc
from bokeh.models import (GMapPlot, GMapOptions, WMTSTileSource, Circle, DataRange1d, PanTool, ResetTool, SaveTool, WheelZoomTool, BoxZoomTool, BoxSelectTool)#ZoomInTool, ZoomOutTool, 
#from bokeh.tile_providers import STAMEN_TONER
#from bokeh.tile_providers import CARTODBPOSITRON
#from pyproj import Proj, transform
import geocoder

app = Flask(__name__)

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')

@app.route('/temp')
def temp():
  return render_template('temp.html')

@app.route('/testing')
def testing():
  return render_template('testing-ml.html')

@app.route('/bokeh')
def bokeh():
  return render_template('bokeh.html')

verbose = 0
df = pd.DataFrame()
city = ""

@app.route('/graph', methods = ['POST'])
def graph():
	global df
	global city
	boro = request.form['boro']
	print("The requested borough is '" + boro.upper() + "'")
	plot = request.form['plot']
	print("The requested plot is '" + plot + "'")
	aftermonth = request.form['month']
	print("The requested month is '" + aftermonth + "'")
	afteryear = request.form['year']
	afterdate = r"'"+afteryear+'-'+aftermonth+r"-01'"#r"'2016-01-01'"
	#'''
	cache_dir = 'cache'
	if not os.path.exists(cache_dir):
	    os.mkdir(cache_dir)

	@checkpoint(key=lambda args, kwargs: urllib.parse.quote(args[0]) + '_' + str(args[1]) + '.p', work_dir=cache_dir, refresh=True)  #.parse'''
	def get_data(city, rows, where,date):
	    params = { #'format'        :'json', 
		       '$order':		 'created_date',
				'$limit':         rows, 
				'$where':		 'created_date>='+date, #r"created_date>='2016-01-01'",#'created_date%3E%3D-%272016-01-01%27',#>='2016-01-01'',# between \'2016-01-01T00:00:00\' and 2016-10-01T00:00:00', #"created_date in('2016')",
				'$select':		'created_date,closed_date,agency,incident_zip,complaint_type,descriptor,latitude,longitude',
		       'city' :          city}#,
		        #'$where' :         where}
	    print('making API request...')
	    result = requests.get(url, params=params)
	    print('API request complete.')
	    return result
	
	if boro.lower() == 'manhattan':# or 'nyc':
		city = 'NEW YORK'
	else:
		city = boro.upper()

	calls = 50000#100000
	agencies = ("DEP","DOB","DOT","HPD","NYPD","DSNY","FDNY","DPR")#, "DOHMH")#,"DHS")

	agencyList = ','.join('"%s"' % x for x in agencies)
	print(agencyList)
	print('getting data from '+city)
	r = get_data(city,calls,'agency in('+agencyList+')',afterdate)

	df = pd.read_json(r.text, convert_dates=True)
	if verbose>1: print(df.head());print(len(df));print(df.columns.tolist())
	#clean up database
	collist = ['created_date', 'closed_date','agency', 'incident_zip','complaint_type','descriptor', 'latitude', 'longitude']
	newcols = ['created', 'closed', 'agency','zip', 'complaint', 'description','lat', 'long']
	df = df[collist]
	#rename cols
	df.rename(columns=dict(zip(collist, newcols)), inplace=True)
	#print(df.columns[0])
	df = df.dropna()
	if verbose>1: print(df.head(5))
	
	agencydict = {"DEP": 'cyan',"DOB": 'red',"DOT": 'blue',"HPD": 'yellow',"NYPD": 'black',"DSNY": 'magenta',"FDNY": 'orange',"DPR": 'green', "DOHMH": 'gray', "DHS": 'gray'}
	
	startdate = df.created.iloc[0]
	#print(startdate.year, startdate.month, startdate.day)
	startdate = datetime.datetime.strptime(startdate, '%Y-%m-%dT%H:%M:%S.%f')
	if verbose>0: print('%d-%d-%d' % (startdate.month, startdate.day, startdate.year))
	startdate = str(startdate.month)+'/'+str(startdate.day)+'/'+str(startdate.year)
	enddate = df['created'].iloc[-1]
	enddate = datetime.datetime.strptime(enddate, '%Y-%m-%dT%H:%M:%S.%f')
	enddate = str(enddate.month)+'/'+str(enddate.day)+'/'+str(enddate.year)
	#if verbose>0: print(enddate); print(max(df.created)); print(min(df.created))
	print(startdate, enddate)
	
	#y = df['lat'].tolist()
	#print(y)
	#x = df['long'].tolist()
	df['created'] = pd.to_datetime(df.created, infer_datetime_format=True, errors='coerce')
	df['created'] = df.created.dt.date
	countdf = df[['created','agency',]].groupby(['created','agency'], as_index=False).size()
	unstacked = countdf.unstack(level=-1, fill_value=0)
	if verbose>0: print(unstacked)
	unstacked['Date']=[datetime.datetime.combine(d, datetime.datetime.min.time()) for d in unstacked.index.tolist()]#columns['Date']
    #x = np.array(x, dtype=np.datetime64)
    #month = x[0] - np.timedelta64(30,'D')
    #x=x[x>=month]
    #y=y[0:len(x)]
	print('starting to graph')
	title = boro.upper() +' 311 calls  from '+startdate+' to '+enddate#routed to '+agency+'
	defaults.plot_width = 450
	defaults.plot_height = 400

	if plot == 'multi':
		p = figure(x_axis_type="datetime",plot_width=1000, plot_height=800, title=title)# title=title)
		p.grid.grid_line_alpha = 0.5
		p.xaxis.axis_label = 'Date (m-d)'
		p.yaxis.axis_label = 'Number of calls per day'
		p.ygrid.band_fill_color = "olive"
		p.ygrid.band_fill_alpha = 0.1
		for k,v in unstacked.items():
			if k in agencyList:
				p.line(x=unstacked['Date'].tolist(), y=unstacked[k].tolist(), color=agencydict[k], legend=k)	
		#p.line(x, y, color='navy')
		#p.circle(x, y, color='navy')
		#p = TimeSeries(unstacked, x_mapper_type='datetime', xlabel='Date', legend=True, title=title, ylabel='Number of calls per day')
		script, div = components(p)
		return render_template('graph.html', script=script, div=div)

	elif plot == 'grid':
		scatter = []
		for k,v in unstacked.items():#columns.items():#range(len(agencies)):
			if verbose>0: print(k)#agencies[i])
			if k in agencyList:#!= 'Date':
				scatter.append(Scatter(unstacked, x='Date', y=k, color=agencydict[k]))#xyvalues
		'''
		for index, row in df.iterrows():
			print('.', end="")
			p.scatter(row['long'],row['lat'],fill_color=agencydict[row['agency']], line_width=0.0)#, legend="bottom_right")
		'''
		print('starting to render')
		g = gridplot([s for s in scatter], ncols=4, title=title)
		script, div = components(g)
		return render_template('graph.html', script=script, div=div)
		#output_file('templates/gridplots.html', title=title)
		#show(g)
		#return render_template('gridplots.html')
	#'''
	elif plot == 'pockets':
		return redirect('/pockets')#, df=df)#**locals())
		
	#'''
@app.route('/hpd')
def hpd():
	hpdf = pd.read_csv('hpd.csv')
	print(hpdf.head())
	print(len(hpdf))
	#print(hpdf.long.tolist())
	map_options = GMapOptions(lat=40.80, lng=-73.94, map_type="roadmap", zoom=13, styles="""
[{"featureType":"administrative","elementType":"all","stylers":[{"visibility":"on"},{"lightness":33}]},{"featureType":"landscape","elementType":"all","stylers":[{"color":"#f2e5d4"}]},{"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#c5dac6"}]},{"featureType":"poi.park","elementType":"labels","stylers":[{"visibility":"on"},{"lightness":20}]},{"featureType":"road","elementType":"all","stylers":[{"lightness":20}]},{"featureType":"road.highway","elementType":"geometry","stylers":[{"color":"#c5c6c6"}]},{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#e4d7c6"}]},{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#fbfaf7"}]},{"featureType":"water","elementType":"all","stylers":[{"visibility":"on"},{"color":"#acbcc9"}]}]
""") #do NOT change zoom to 12, map won't load!
	plot = GMapPlot(
	    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options, plot_width=600, plot_height=750, 
	    api_key=API_KEY #, tools=TOOLS #title=title, tools=[hover]
	)
	source = ColumnDataSource(data=dict(x=[], y=[]))
	source.data = dict(
		x = hpdf.long.tolist(), 
		y = hpdf.lat.tolist()
	)
	circle = Circle(x='x', y='y', size=5, fill_color="orange", fill_alpha=0.2, line_color=None)
	plot.add_glyph(source, circle)
	plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())#, hover)
	script, div = components(plot)
	return render_template('hpd.html', script=script, div=div)

@app.route('/pockets')
def pockets():
	global df
	global city
	'''
	if city == "NEW YORK":
		borough = 'MANHATTAN'
	else: borough = city
	parms={'$select': 'boro,housenumber,streetname,zip', 'boro': borough}
	url_hpd = "https://data.cityofnewyork.us/resource/sc6a-36xc.json"
	print('requesting HPD data...')
	hpdr = requests.get(url_hpd, params=parms)
	print('request complete, now reading json...')
	hpdf=pd.read_json(hpdr.text, dtype={'zip': 'str'})
	hpdf['address'] = hpdf.housenumber+' '+ hpdf.streetname + ', ' + city# hpdf.boro + ', ' + hpdf.zip
	hpdf.dropna()
	#url_geo = 'https://maps.googleapis.com/maps/api/geocode/json'
	for i,row in hpdf.iterrows():#_using.iterrows():
		#geops = {'sensor': 'false', 'address': row['address'], 'key': API_KEY_GEO}
		#geor = requests.get(url_geo, params=geops)
		#georesponse = geor.json()#['results']
		print(row['address'])
		g = geocoder.osm(row['address'])
		print(g.osm)
		if g.osm:
			hpdf['long'] = g.osm["x"]
			hpdf['lat'] = g.osm["y"]
		else: print('no osm')
		#if georesponse['results']:
			#print(georesponse)
			#latlong = georesponse['results'][0]['geometry']['location']#['lat']
			#hpdf['lat'] = latlong['lat']
			#hpdf['long'] = latlong['lng']
		#else: print('no georesponse')
	'''
	#hpdf = pd.read_csv('hpd.csv')
	#print(hpdf.head())
	#print(len(hpdf))
	
	#desc = Div(text=open(join(dirname(__file__), "templates/description.html")).read(), width=800)
	slider = Slider(start=0, end=len(df)-1, value=1, step=len(df)/100, title="time period")#, callback=update)
	
	# Create Column Data Source that will be used by the plot
	source = ColumnDataSource(data=dict(x=[], y=[], created=[], agency=[]))#, title=[], year=[], revenue=[], alpha=[]))
	#hpd_source = ColumnDataSource(data=dict(long=hpdf.long, lat=hpdf.lat))
	hover = HoverTool(tooltips=[
		#("Neighborhood", "@hood"),
		("# calls", "@created"),
		("agencies", "@agency"),
		("location", "(@x, @y)")
	])
	pan = PanTool()
	wheel_zoom = WheelZoomTool()
	#box_zoom = BoxZoomTool()
	box_select = BoxSelectTool()
	#zoom_in = ZoomInTool()
	#zoom_out = ZoomOutTool()
	#save = SaveTool()
	#reset = ResetTool()
	TOOLS = "pan, box_zoom, wheel_zoom, reset, save, hover"
	'''
	NYC = x_range, y_range = ((-8242000,-8210000), (4965000,4990000))

	plot_width  = int(750)
	plot_height = int(plot_width//1.2)

	def base_plot(tools='pan,wheel_zoom,reset, hover',plot_width=plot_width, plot_height=plot_height, **plot_args):
	    p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,
	        x_range=x_range, y_range=y_range, outline_line_color=None,
	        min_border=0, min_border_left=0, min_border_right=0,
	        min_border_top=0, min_border_bottom=0, **plot_args)
	    
	    p.axis.visible = False
	    p.xgrid.grid_line_color = None
	    p.ygrid.grid_line_color = None
	    return p
	    
	options = dict(line_color=None, fill_color='red', size=10)
	# transform lng/lat to meters
	def to_meters(lng,lat):
		from_proj = Proj(init="epsg:4326")
		to_proj = Proj(init="epsg:3857")
		x, y = transform(from_proj, to_proj, lng, lat)#-73.764214, 45.542642)
		return x,y

	'''
	maplat = df.lat.mean() #40.81
	maplong = df.long.mean() #-73.94
	map_options = GMapOptions(lat=maplat, lng=maplong, map_type="roadmap", zoom=13, styles="""
[{"featureType":"administrative","elementType":"all","stylers":[{"visibility":"on"},{"lightness":33}]},{"featureType":"landscape","elementType":"all","stylers":[{"color":"#f2e5d4"}]},{"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#c5dac6"}]},{"featureType":"poi.park","elementType":"labels","stylers":[{"visibility":"on"},{"lightness":20}]},{"featureType":"road","elementType":"all","stylers":[{"lightness":20}]},{"featureType":"road.highway","elementType":"geometry","stylers":[{"color":"#c5c6c6"}]},{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#e4d7c6"}]},{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#fbfaf7"}]},{"featureType":"water","elementType":"all","stylers":[{"visibility":"on"},{"color":"#acbcc9"}]}]
""") #do NOT change zoom to 12, map won't load!
	plot = GMapPlot(
	    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options, plot_width=800, plot_height=750, 
	    api_key=API_KEY #, tools=TOOLS #title=title, tools=[hover]
	)
	#plot = figure(plot_height=600, plot_width=700, title="", toolbar_location=None, tools=[hover])#p
	#plot = base_plot()
	#plot.add_tile(STAMEN_TONER)
	#plot.circle(x="x", y="y", source=source, size=10, line_color=None, fill_color="red")#**options, fill_alpha="alpha")#p
	circle = Circle(x="x", y="y", size=10, fill_color="red", fill_alpha=0.9, line_color=None)
	plot.add_glyph(source, circle)
	#plot.add_glyph(hpd_source, Circle(x='long', y='lat', size=6, fill_color="orange", fill_alpha=0.4, line_color=None))
	plot.add_tools(pan, wheel_zoom, box_select, hover)#box,zoom, reset don't work...
	def update(start=0):#, end=len(df), col='Close'):
		startdate = df.created[start]
		#startdate = datetime.datetime.strptime(startdate, '%Y-%m-%dT%H:%M:%S.%f')
		enddate = startdate + relativedelta(months=1)
		if enddate < max(df.created):
			endpt = df[df.created >= enddate].index[0]
		else: 
			endpt = df.index[-1]
			enddate = df.created[endpt]
		ending = str(enddate.month)+'/'+str(enddate.day)+'/'+str(enddate.year)
		starting = str(startdate.month)+'/'+str(startdate.day)+'/'+str(startdate.year)
		#plot = figure(tools=TOOLS, title='problematic areas between '+starting+' and '+ending)#, x_axis_label="date", x_axis_type="datetime")
		plot.title.text = 'problematic areas in '+city+' between '+starting+' and '+ending
		pockets = df[start:endpt].groupby(['lat','long']).agg({'created': 'count', 'agency': lambda x: ', '.join(x)})
		counterizer = lambda x: Counter(x)
		lister = lambda x: x.split(", ")
		uncounter = lambda x: sorted([(v,k) for (k,v) in x.items()], reverse=True)#sorted(list(x.items()))#
		nicelist = lambda x: [(' '+k+': '+str(v)) for (v,k) in x]
		pockets['agency'] = pockets.agency.apply(lister).apply(counterizer).apply(uncounter).apply(nicelist)#pockets.agency.apply(Counter(), axis=1)
		#print(pockets.agency)
		thresh = 2*(endpt-start)/1000#2*len(pockets)/1000
		#print(start, endpt, len(pockets), thresh)
		#alltrans = [to_meters(y,x) for x,y in pockets[pockets.created>thresh].index]
		source.data = dict(
				x=[y for x,y in pockets[pockets.created>thresh].index],
				y=[x for x,y in pockets[pockets.created>thresh].index],
				#x = alltrans[:][0],
				#y = alltrans[:][1],
				created = pockets[pockets.created>thresh].created,
				agency = pockets[pockets.created>thresh].agency,
		)
		#plot.scatter(x, y)
		#show(plot)

	controls = [slider]#reviews, boxoffice, genre, min_year, max_year, oscars, director, cast, x_axis, y_axis]
	for control in controls:
		control.on_change('value', lambda attr, old, new: update())

	sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example
	inputs = widgetbox(*controls, sizing_mode=sizing_mode)
	l = layout([
		#[desc],
		[inputs, plot],
	], sizing_mode=sizing_mode)

	update()  # initial load of the data
	curdoc().add_root(l) #l~row(inputs, plot, width=800)
	curdoc().title = "311 Data"
	#show(widgetbox(slider))		
	#interact(update, start=(0,len(df)-1))#, end=(1,len(df)),col=('Open','Close'))
	script,div = components(plot) #{"p": plot, "slider": slider})#vform(slider)})
	return render_template('pockets.html', script=script, div=div)
	#return render_template('pockets.html')



@app.route('/map')
def map():
	'''
	global df
	#---------------------population data from census database
	pdata = requests.get(url_census)
	pdf = pd.read_json(pdata.text)
	if verbose>1: print(pdf.columns.tolist())
	coll = ['geographic_area_neighborhood_tabulation_area_nta_code', 'geographic_area_neighborhood_tabulation_area_nta_name', 'total_population_2010_number']
	newl = ['nta_code','hood','population']
	pdf = pdf[coll]
	pdf.rename(columns=dict(zip(coll, newl)), inplace=True)
	pdf.set_index('hood')
	if verbose>0: print(pdf.head(15))
	pX = pdf.as_matrix()

	#---------------------load GeoJSON file containing neighborhood polygons
	with open('static/nynta.geojson', 'r') as f:
	    js = json.load(f)
	polygons = {}
	for feature in js['features']:
		#print(feature['properties']['NTAName']) #if nynta, 'neighborhoods' for nychoods
		polygons[feature['properties']['NTAName']] = shape(feature['geometry'])
		
	#print(polygons.items())
	def which_neighb(latlong):
		''''''Outputs NY neighborhood name given (lat,long).''''''
		
		# construct point based on lat/long returned by the geocoder geocoder
		#print('about to convert to point...')
		point = Point(latlong[0],latlong[1])
		# check each polygon to see if it contains the point
		for neighb,polygon in polygons.items():
			#print(neighb)
			if polygon.contains(point):
				return neighb
		else: return None#'Not in New York!'

	#pool = multiprocessing.Pool()
	#neigh = pool.map(which_neighb, df[['long','lat']].apply(tuple, axis=1))
	neigh = [which_neighb(x) for x in df[['long','lat']].apply(tuple, axis=1)]
	#pool.close()
	#pool.join()
	#print(neigh)
	f.close()

	df['hood'] = neigh
	#df['hood'] = df.apply(which_neighb(df['long'],df['lat']), axis = 1)#np.vectorize(which_neighb(df['lat'],df['long']))# 
	#df = df[df['hood' != 'Not in New York!']]
	df = df.dropna()
	if verbose>1: print(df.head())

	#use complaints instead of agency below for more granularity
	grouped = df[['created','agency','hood','lat','long']].groupby(['hood','agency'], as_index=False).agg({'lat': 'mean', 'long':'mean','created': 'count'})

	cgrouped = df[['created','agency','complaint','hood','lat','long']].groupby(['hood','agency','complaint'], as_index=False).agg({'lat': 'mean', 'long':'mean','created': 'count'})
	if verbose>1: print(grouped.head(30))
	#----------------------- groupings to merge
	agencygrp = df[['created','agency']].groupby(['agency'], as_index=False).count()
	if verbose>1: print(agencygrp)
	ttlcalls = sum(agencygrp.created)
	if verbose>1: print(ttlcalls)

	hoodgrp = df[['created','hood']].groupby(['hood'], as_index=False).count()
	if verbose>1: print(hoodgrp) #print(hoodgrp.ix[hoodgrp.hood == "West Village", 'created'])

	popgrp = hoodgrp.merge(pdf[['hood','population']], on='hood')
	popgrp.rename(columns={'hood_x':'hood'}, inplace=True)
	#ratio gives #calls/ppl in neighborhood
	popgrp = popgrp.assign(ratio = lambda x: (x.created / x.population))
	if verbose>1: print('this is population group:'); print(popgrp)
	#-------------------- original agency version:
	df0 = grouped.merge(agencygrp[['agency','created']], on='agency')
	df0.rename(columns={'created_x':'created'}, inplace=True)
	df0.rename(columns={'created_y':'agencyttl'}, inplace=True)
	#print(df0)
	df1 = df0.merge(hoodgrp[['hood','created']], on='hood')
	df1.rename(columns={'created_x':'created'}, inplace=True)
	df1.rename(columns={'created_y':'hoodttl'}, inplace=True)
	#print(df1)
	#df2 = df1.assign(ratio = lambda x: ((x.created/x.hoodttl) / (x.agencyttl/ttlcalls)))
	df2 = df1.assign(ratio = lambda x: ((x.created) / (x.agencyttl/ttlcalls)))
	df3 = df2[['hood','agency','ratio','created','lat','long']]#
	df4 = df3.sort(['hood','ratio'],ascending=False).groupby('hood').head(3)
	#print(df4)
	#--------------------- complaint version:
	cdf0 = cgrouped.merge(agencygrp[['agency','created']], on='agency')
	cdf0.rename(columns={'created_x':'created'}, inplace=True)
	cdf0.rename(columns={'created_y':'agencyttl'}, inplace=True)
	#print(df0)
	cdf1 = cdf0.merge(hoodgrp[['hood','created']], on='hood')
	cdf1.rename(columns={'created_x':'created'}, inplace=True)
	cdf1.rename(columns={'created_y':'hoodttl'}, inplace=True)
	if verbose>1: print(df1)
	#ratio gives probability of a complaint type(agency) in particular neighborhood
	#df2 = df1.assign(ratio = lambda x: ((x.created/x.hoodttl) / (x.agencyttl/ttlcalls)))
	cdf2 = cdf1.assign(ratio = lambda x: ((x.created) / (x.agencyttl/ttlcalls)))
	#print(df2.nlargest(3, columns='ratio'))
	df3 = df2[['hood','agency','ratio','created','lat','long']]
	cdf3 = cdf2[['hood','agency','complaint','ratio','created','lat','long']]#
	if verbose>1: print(df3)
	#df4 = df3['ratio'].groupby(level=0, group_keys=True)
	cdf4 = cdf3.sort(['hood','ratio'],ascending=False).groupby('hood').head(3)
	if verbose>0: print(cdf4.head())

	cX = cdf4.as_matrix(columns=df4.columns[2:])
	#print(len(cX))
	rcmax = max(cX[:,1])
	cdf4['ratio'] = cdf4['ratio'].apply(lambda x: x/rcmax)
	cxf = cdf4.as_matrix()
	X = df4.as_matrix(columns=df4.columns[1:])
	#print(len(X))
	rmax = max(X[:,1])
	df4['ratio'] = df4['ratio'].apply(lambda x: x/rmax)
	xf = df4.as_matrix()

	#use complaints instead of agency below for more granularity
	dgrouped = df[['created','description','hood']].groupby(['hood','description'], as_index=False).agg({'created': "count"})
	print(dgrouped.head(30))

	#----------------------complaint descriptions
	descgrp = df[['created','description']].groupby(['description'], as_index=False).count()
	#print(descgrp)
	#ttlcalls = sum(complgrp.created)
	print('total # of calls:',ttlcalls)
	dhoodgrp = df[['created','hood']].groupby(['hood'], as_index=False).count()
	if verbose>1: print(hoodgrp)

	ddf0 = dgrouped.merge(descgrp[['description','created']], on='description')
	ddf0.rename(columns={'created_x':'created'}, inplace=True)
	ddf0.rename(columns={'created_y':'complttl'}, inplace=True)
	#print(df0)
	ddf1 = ddf0.merge(hoodgrp[['hood','created']], on='hood')
	ddf1.rename(columns={'created_x':'created'}, inplace=True)
	ddf1.rename(columns={'created_y':'hoodttl'}, inplace=True)
	if verbose>1: print(df1.head())
	#ratio gives probability of a particular complaint in particular neighborhood
	ddf2 = ddf1.assign(ratio = lambda x: ((x.created/x.hoodttl) / (ttlcalls)))
	#print(df2.nlargest(3, columns='ratio'))
	ddf3 = ddf2[['hood','description','ratio']]
	#sort by top 3 complaint descriptions
	ddf4 = ddf3.sort(['hood','ratio'],ascending=False).groupby('hood').head(3)
	dX = ddf4.as_matrix()

	#-----------------------insert calcs into geojson
	import pygeoj
	nf = pygeoj.load("static/nynta.geojson")
	for i,feature in enumerate(nf):
		count = 1
		notFound = 1
		for i,row in enumerate(xf):
			#print(row)	
			#find neighborhood
			if feature.properties['NTAName'] == row[0] and notFound:
				#update property
				key = "top"+str(count)
				key2 = "ag"+str(count)
				key3 = "count"+str(count)
				key4 = "compl"+str(count)
				key5 = "descr"+str(count)
				#print('updating property...with '+key4)
				feature.properties[key] = row[2]	#agency ratios
				feature.properties[key2] = row[1]	#agency
				feature.properties[key3] = row[3]	#counts in agency
				feature.properties[key4] = cxf[i][2]	#complaint
				#print(dX[np.where(dX == row[0])[0]][count-1][1])
				feature.properties[key5] = dX[np.where(dX == row[0])[0]][count-1][1]	#description
				#print(pdf.get_value(pdf.index.get_loc(feature.properties['NTAName']),'population'))
				rows, columns = np.where(pX==row[0])
				#first_idx = sorted([r for r, c in zip(rows, columns) if c == 0])[0]
				#print(rows, columns)
				if count ==1 and rows:
					feature.properties['population'] = pX[np.where(pX == row[0])[0]][0][2]
					#print(row[0],hoodgrp[hoodgrp.hood == row[0]].get_value(hoodgrp[hoodgrp.hood == row[0]].index[0],'created'))
					tcalls = hoodgrp[hoodgrp.hood == row[0]].get_value(hoodgrp[hoodgrp.hood == row[0]].index[0],'created')
					feature.properties['ttlcalls'] = int(tcalls)
				count += 1
				if count > 3:
					count = 1
					notFound = 0
					#break
			#else: 
				#print('deleting feature...')
				#feature.properties["ag1"] = " "
				#del nf[i]
	nf.save('static/nynta_var.geojson')
	os.system('echo "var myGeo = $(cat static/nynta_var.geojson)" > static/nynta_var.geojson')
	'''
	return render_template('map.html')  

if __name__ == '__main__':
  app.run(port=33507)#localhost=33507)
#----------------------------------------------------------------------------------

