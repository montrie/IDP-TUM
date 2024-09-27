import os
import sys

# make this file visible
sys.path.insert(2, os.path.dirname(os.path.abspath(__file__)))

# define root parent directory so that data/ is also visible
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the following variables define the column names of the csv file containing the detector measurements from whic
# we want to match the detector locations to a street network. These names are necessary to set up the 'GPSConfig'
# object needed to run the 'STMATCH' algorithm in match.py
ID = 'detid'                    # defines the column name containing the detector ids
GEOM = 'geometry'               # column name containing the geometry of the network edges
X = 'lon'                       # column name containing the x coordinates
Y = 'lat'                       # column name containing the y coordinates
GPS_POINT = True                # defines whether the map matching algorithm matches gps coordinates to a network
MATCHED_OUTPUT = "matched.csv"  # defines the name of the csv file containing the map matched detector locations

#the following variables define the input parameters for osmnx.graph_from_place() in connect_detectors.py
OSMNX_NETWORK_PLACE = "München, Bayern, Deutschland"  # The place name of the network that is downloaded
OSMNX_NETWORK_TYPE = "drive"                          # The network type (valid: "all", "all_public", "bike", "drive", "drive_service", "walk")
OSMNX_SIMPLIFY = True                                 # Defines whether the network simplification process is performed
