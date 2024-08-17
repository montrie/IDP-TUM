var wms_layers = [];


        var lyr_OSM_0 = new ol.layer.Tile({
            'title': 'OSM',
            //'type': 'base',
            'opacity': 1.000000,
            
            
            source: new ol.source.XYZ({
    attributions: ' ',
                url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
            })
        });
var format_OSMnxedges_1 = new ol.format.GeoJSON();
var features_OSMnxedges_1 = format_OSMnxedges_1.readFeatures(json_OSMnxedges_1, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_OSMnxedges_1 = new ol.source.Vector({
    attributions: ' ',
});
jsonSource_OSMnxedges_1.addFeatures(features_OSMnxedges_1);
var lyr_OSMnxedges_1 = new ol.layer.Vector({
                declutter: false,
                source:jsonSource_OSMnxedges_1, 
                style: style_OSMnxedges_1,
                popuplayertitle: "OSMnx edges",
                interactive: true,
                title: 'OSMnx edges'
            });

lyr_OSM_0.setVisible(true);lyr_OSMnxedges_1.setVisible(true);
var layersList = [lyr_OSM_0,lyr_OSMnxedges_1];
lyr_OSMnxedges_1.set('fieldAliases', {'fid': 'fid', 'u': 'u', 'v': 'v', 'key': 'key', 'osmid': 'osmid', 'name': 'name', 'highway': 'highway', 'maxspeed': 'maxspeed', 'oneway': 'oneway', 'reversed': 'reversed', 'length': 'length', 'prior_flow': 'prior_flow', 'flow': 'flow', 'from': 'from', 'to': 'to', 'lanes': 'lanes', 'ref': 'ref', 'bridge': 'bridge', 'access': 'access', 'tunnel': 'tunnel', 'width': 'width', 'junction': 'junction', 'est_width': 'est_width', });
lyr_OSMnxedges_1.set('fieldImages', {'fid': '', 'u': '', 'v': '', 'key': '', 'osmid': '', 'name': '', 'highway': '', 'maxspeed': '', 'oneway': '', 'reversed': '', 'length': '', 'prior_flow': '', 'flow': '', 'from': '', 'to': '', 'lanes': '', 'ref': '', 'bridge': '', 'access': '', 'tunnel': '', 'width': '', 'junction': '', 'est_width': '', });
lyr_OSMnxedges_1.set('fieldLabels', {'fid': 'no label', 'u': 'no label', 'v': 'no label', 'key': 'no label', 'osmid': 'no label', 'name': 'no label', 'highway': 'no label', 'maxspeed': 'no label', 'oneway': 'no label', 'reversed': 'no label', 'length': 'no label', 'prior_flow': 'no label', 'flow': 'no label', 'from': 'no label', 'to': 'no label', 'lanes': 'no label', 'ref': 'no label', 'bridge': 'no label', 'access': 'no label', 'tunnel': 'no label', 'width': 'no label', 'junction': 'no label', 'est_width': 'no label', });
lyr_OSMnxedges_1.on('precompose', function(evt) {
    evt.context.globalCompositeOperation = 'normal';
});