import datetime
import glob
import os
import logging
import qgis.utils

from time import sleep
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor
from qgis._gui import QgsMapCanvas
from qgis.core import *
from qgis.utils import iface
from qgis.PyQt import QtGui
from shapely import GeometryType
from config import ROOT_DIR
from qgis.core import QgsVectorLayer, QgsProject, QgsApplication, QgsVectorFileWriter
import json, tempfile, os
from PyQt5.QtCore import Qt

edgesData = ROOT_DIR + "/data/network/edges.shp"
nodesData = ROOT_DIR + "/data/network/detector_nodes.gpkg"
mergedDataRoot = ROOT_DIR + "/data/merged_data/"
projectDataRoot = ROOT_DIR + "/data/project_data/"
initial_width = 1
factor = 2/3
trafficRanges = [[(0.0, 0.9), 'No Data', QtGui.QColor('#ff70a0'), initial_width],
                 [(1.0, 100.0), 'Very Low Traffic', QtGui.QColor('#006400'), initial_width * (1 + factor*1)],
                 [(100.1, 200.0), 'Low Traffic', QtGui.QColor('#55d555'), initial_width * (1 + factor*2)],
                 [(200.1, 300.0), 'Normal Traffic', QtGui.QColor('#f5ff09'), initial_width * (1 + factor*3)],
                 [(300.1, 400.0), 'High Traffic', QtGui.QColor('#ffa634'), initial_width * (1 + factor*4)],
                 [(400.1, 1000.0), 'Very High Traffic', QtGui.QColor('#ff2712'), initial_width * (1 + factor*5)]]

prefixPath = r'/usr'  # TODO: change prefix path to your QGIS root directory
QgsApplication.setPrefixPath(prefixPath, True)
qgs = QgsApplication([], False)  # second argument disables the GUI
qgs.initQgis()  # load providers
project = QgsProject.instance()  # save a reference to the current instance of QgsProject

# Adding Map
tms = '    crs=EPSG:3857&format&type=xyz&url=https://tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png&zmax=19&zmin=0'
rlayer = QgsRasterLayer(tms, 'OSM', 'wms')
if rlayer.isValid():
    logging.info("Base Map loaded!")
    project.addMapLayer(rlayer)
else:
    logging.info(rlayer.error().summary())


def add_color(traffic_range: (float, float),
              label: str,
              color: QColor,
              line_width: float,
              geom_type: GeometryType,
              opacity: float = 1.0,
               ) -> QgsRendererRange:
    """
    Colorize the detector data according to the measured traffic flow

    :param traffic_range: Lower and upper bounds for heaviness of traffic
    :param label: Label the amount of traffic flow
    :param color: Color corresponding to the measured traffic
    :param geom_type: GeometryType of the loaded detector data
    :param opacity: Opacity of the corresponding points
    :param line_width: thickness of line
    :return: QgsRendererRange instance
    """
    (lower, upper) = traffic_range
    symbol = QgsSymbol.defaultSymbol(geom_type)
    line_layer = QgsSimpleLineSymbolLayer()
    line_layer.setWidth(line_width)
    symbol.changeSymbolLayer(0, line_layer)
    symbol.setColor(color)
    symbol.setOpacity(opacity)

    return QgsRendererRange(lower, upper, symbol, label)


def add_vector_layer(layer: QgsVectorLayer):
    """
    Adds the given QgsVectorLayer to the current QGIS project if the layer is valid. Otherwise, print the error summary

    :param layer: A QgsVectorLayer object
    """
    if layer.isValid():
        logging.info(layer.name() + " loaded successfully!")
        project.addMapLayer(layer)
    else:
        logging.info(layer.error().summary())


def mapAndPoint():
    """
    Define a base map using OSM and load previously pulled detector location data. Color the detector points
    according to the measured traffic flow and map the points on top of the base map. Save the resulting .qgs file
    in project_data.
    """

    # Adding Points 14748 osmn point, 21bin osmn edges,  36 bin testlayer
    # HINT: ':=' available for python >= 3.8 -> compatibility issues?
#    if not (merged_points_data_csv := max(glob.glob(mergedDataRoot + "*.csv"), key=os.path.getctime)):
#        logging.error("No files found in {}".format(mergedDataRoot))
    # include line below to use map matched detector locations
#    merged_points_data_csv = os.path.join(ROOT_DIR, "data/network/matched.csv")
    # matched.csv uses ';' as delimiter -> check if we use the matched detector locations as the csvlayer
#    delim = ';' if "matched.csv" in merged_points_data_csv[-11:] else ','
#    options = '?delimiter={}&xField=lon&yField=lat&crs=epsg:4326'.format(delim)
#    uri = "file:///{}{}".format(merged_points_data_csv, options)

    gpkg_edgelayer = QgsVectorLayer(nodesData + "|layername=edges", "OSMnx edges", "ogr")
#    csvlayer = QgsVectorLayer(uri, "Points", "delimitedtext")

# HINT: the commented lines of code can be used to help in setting up visualization of the detectors directly
#    add_vector_layer(csvlayer)
    add_vector_layer(gpkg_edgelayer)
    project.setCrs(QgsCoordinateReferenceSystem('EPSG:3857'), True)

    # create data-driven opacity property
    symbol = QgsSymbol.defaultSymbol(gpkg_edgelayer.geometryType())
    opacity_property = QgsProperty.fromExpression('if("prior_flow" = true, 100, 50)')
    symbol.setDataDefinedProperty(QgsSymbol.Property.Opacity, opacity_property)

    # create a rule based renderer using the created symbol
    edge_renderer = QgsRuleBasedRenderer(symbol)
    root_rule = edge_renderer.rootRule()

    # add a range in case we have any NULL flow values left
    range_list = [[('NULL', 'NULL'), "No data", QtGui.QColor('#D3D3D3'), 0.66]] + trafficRanges
    # reverse the range_list, in theory higher traffic flow is visualized last now
    range_list = range_list[::-1]
    for color_range in range_list:
        rule = root_rule.children()[0].clone()
        rule.setLabel(color_range[1])
        # define the rule
        expression = '"flow" >= {} AND "flow" <= {}'.format(color_range[0][0], color_range[0][1]) \
            if color_range[1] != "No data" \
            else '"flow" IS NULL'
        rule.setFilterExpression(expression)
        rule.symbol().setColor(color_range[2])
        rule.symbol().symbolLayer(0).setWidth(color_range[3])
        for i in range(rule.symbol().symbolLayerCount()):
            layer = rule.symbol().symbolLayer(i)
            layer.setDataDefinedProperty(QgsSymbolLayer.PropertyOpacity, opacity_property)
        root_rule.appendChild(rule)

    root_rule.removeChildAt(0)  

    gpkg_edgelayer.setRenderer(edge_renderer)
    gpkg_edgelayer.triggerRepaint()

    # TODO: How to center QGIS on the OSMnx edges layer? I cant figure it out :)
    #temp = project.mapLayersByName("OSMnx edges")[0]
    #canvas = qgis.utils.iface.mapCanvas()
    #canvas.setExtent(temp.extent())
    #canvas.refresh()

    # save the created QGIS project
    project.write(filename=os.path.join(projectDataRoot, "mapAndPoint.qgs"))
    updateDataforWebsite(gpkg_edgelayer)


def updateDataforWebsite(gpkg_edgelayer):
    """
    Updates the JavaScript file containing the information about all features of the edges in gpkg_edgelayer
    :param gpkg_edgelayer: A QgsVectorLayer that contains the visualization of the network to be displayed on the website
    """
    # create temporary geojson file path
    (temp_fd, temp_path) = tempfile.mkstemp(suffix=".geojson")
    os.close(temp_fd)
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.fileEncoding = "utf-8"
    options.driverName = "GeoJSON"
    # qgis doesn't allow skipping the transform context by using the parameter name options=options, so we need to pass an empty context
    writer = QgsVectorFileWriter.writeAsVectorFormatV3(gpkg_edgelayer, temp_path, QgsCoordinateTransformContext(), options)

    # read GeoJSON file and save content in a Javascript variable
    with open(temp_path, 'r') as geojson_file:
        geojson_content = geojson_file.read()
        js_content = f"var json_OSMnxedges_1 = {geojson_content};"

    # Write the gpkg data of the js variable into the JavaScript file
    js_path = os.path.join(ROOT_DIR, "layers", "OSMnxedges_1.js")
    with open(js_path, "w") as js_file:
        js_file.write(js_content)

    # Delete temporary folder
    os.remove(temp_path)

    logging.info("Geopackage information converted to JavaScript file, network visualization update saved.")

def layer():
    """
    If projectDataRoot is not empty, delete its files before mapping the detector data onto a base map
    """

    # https://stackoverflow.com/questions/53513/how-do-i-check-if-a-list-is-emptyhttps://stackoverflow.com/questions/53513/how-do-i-check-if-a-list-is-empty
    if os.listdir(projectDataRoot):
        pass
#        delete_files_in_directory(projectDataRoot)
    else:
        logging.info("No files found in the directory.")
    mapAndPoint()


def delete_files_in_directory(directory_path):
    """
    Delete all files in a given directory, ignores sub-directories

    :param directory_path: Absolute path of the directory
    """

    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        logging.info("All files deleted successfully.")
    except OSError:
        logging.error("Error occurred while deleting files.")


def main():
    layer()


if __name__ == '__main__':
    main()

