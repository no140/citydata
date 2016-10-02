//var mymap = L.map('mapid').setView([41.8281, -87.5998], 10);
var mymap = L.map('mapid').setView([40.7831, -73.9512], 13)
L.tileLayer('https://api.mapbox.com/styles/v1/tashazo/cit3a0knx00582wml6l3efq2w/tiles/256/{z}/{x}/{y}?access_token={accessToken}', {
    attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="http://mapbox.com">Mapbox</a>',
            maxZoom: 18,
            id: "mapbox.light",
            accessToken: 'pk.eyJ1IjoidGFzaGF6byIsImEiOiJjaXQzOXZrc3AwdXIzMnBraGcyd2lrcm4yIn0._i-bkyK7sXAqlsxaDO6AUg'
            }).addTo(mymap);

mymap.scrollWheelZoom.disable();

function getColor(d) {
    return d == "DOT" ? '#3182bd' :
	   d == "DEP"  ? '#1c9099' :
	   d == "DOB"  ? '#8856a7' ://'#de2d26' :
	   d == "HPD"  ? '#f03b20' :
	   d == "NYPD"   ? '#636363' :
	   d == "FDNY"   ? '#e6550d' :
	   d == "DSNY"   ? '#c51b8a' :
	   d == "DPR"   ? '#2ca25f' :
		      '#FFEDA0';
}            

function style(feature) {
    return {
        fillColor: getColor(feature.properties['ag1']),
            weight: 3,
            opacity: 1,
            color: getColor(feature.properties['ag2']),//'white',
            dashArray: '3',
            fillOpacity: 0.7
            };
}

function highlightFeature(e) {
    var layer = e.target;

    layer.setStyle({
        weight: 5,
                color: '#666',
                dashArray: '',
                fillOpacity: 0.7
                });

    if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
        layer.bringToFront();
    }
    info.update(layer.feature.properties);
}

function resetHighlight(e) {
    geojson.resetStyle(e.target);
    info.update();
}

var geojson;
// ... our listeners

function zoomToFeature(e) {
    mymap.fitBounds(e.target.getBounds());
}

function onEachFeature(feature, layer) {
    layer.on({
        mouseover: highlightFeature,
                mouseout: resetHighlight,
                click: zoomToFeature
                });
}

geojson = L.geoJson(myGeo, {
    style: style,
    onEachFeature: onEachFeature
    }).addTo(mymap);

//
var info = L.control();
//var timediv = $('#timeid')

info.onAdd = function (map) {
    this._div = L.DomUtil.create('div', 'info'); // create a div with a class "info"
    this.update();
    return this._div;
};

// method that we will use to update the control based on feature properties passed
info.update = function (props) {
    //this._div.innerHTML = '<img src="night.gif" alt="CRIMES_JULY" style="height:100px;">'
    //this._div.innerHTML = '<h4>Share of crimes per hour</h4>' +  (props ?
      //  '<img src="jpegs/freq_polar_' + props.NameCode + '.jpeg" alt="CRIMES_"' + props.NameCode + ' style="height:200px;"><br><center>' + props.community + '<br> Total Crimes: ' + props.NCrimes + '</center>'
	this._div.innerHTML =  (props ?
        '<b><h4>' + props['NTAName'] + '</h4></b>' + 'population: '+props['population']
		+ '<h5>Most commonly called agencies out of ' + props['ttlcalls'] + ' total calls: </h5>' 
				+ '</b>' + props['ag1'] + ': ' + props['count1']
				+ '</b><br />' + props['ag2'] + ': ' + props['count2']
				+ '</b><br />' + props['ag3'] + ': ' + props['count3']
				+ '</b><br /><i><h5>top complaints:</h5></i> ' 
				+ '</b>1) ' + props['compl1'] 
				+ '</b><br />2) ' + props['compl2']
				+ '</b><br />3) ' + props['compl3']
				+ '</b><br /><i><h5>and descriptions:</h5></i> ' 
				+ '</b>1) ' + props['descr1'] 
				+ '</b><br />2) ' + props['descr2']
				+ '</b><br />3) ' + props['descr3']
                                      : 'Hover over a neighborhood');
};

info.addTo(mymap);

var legend = L.control({position: 'bottomright'});
legend.onAdd = function (map) {
    var div = L.DomUtil.create('div', 'info legend'),
        grades = ["DEP","DOB","DOT","HPD","NYPD","DSNY","FDNY","DPR"],//, "DOHMH","DHS"],
        labels = ['Agencies'];

    // loop through our density intervals and generate a label with a colored square for each interval
    for (var i = 0; i < grades.length; i++) {
        div.innerHTML +=
            '<i style="background:' + getColor(grades[i]) + '"></i> ' +
            grades[i] + '<br>';
    }
    return div;
};
legend.addTo(mymap);

//var legend = L.control({position: 'bottomright'});

//legend.onAdd = function (map) {

  //  var div = L.DomUtil.create('div', 'info legend'),
    //    grades = [10, 20, 50, 100, 200, 500, 1000],
      //  labels = ['Crimes per 1000 <br>residents since 2001'];

    // loop through our density intervals and generate a label with a colored square for each interval
        //div.innerHTML += labels.join('<br>') + '<br>';
//    for (var i = 0; i < grades.length; i++) {
  //      div.innerHTML +=
    //    '<i style="background:' + getColor((grades[i] + 1.0)/1000) + '"></i> ' +
      //  grades[i] + (grades[i + 1] ? '&ndash;' + grades[i + 1] + '<br>' : '+');
  //  }

    //return div;
//};

//legend.addTo(mymap);
