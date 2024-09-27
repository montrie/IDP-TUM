import sys
import logging

from fmm import FastMapMatch, STMATCH, Network, NetworkGraph, UBODTGenAlgorithm, UBODT 
from fmm import GPSConfig, ResultConfig, STMATCHConfig, FastMapMatchConfig
from config import ID, GEOM, X, Y, GPS_POINT, MATCHED_OUTPUT

network_dir = "../data/network/"


def match(fpoints: str):
    """
    Match the detector points defined in the file given by fpoints to the street network defined in edges.shp

    :param fpoints: Path to the csv file containing the detector locations to be matched to the network
    """
    ### Read network data
    network = Network(network_dir+"edges.shp","fid","u","v")
    graph = NetworkGraph(network)
   
    ### Create STMATCH model
    model = STMATCH(network, graph)

    ### Define map matching configurations
    k = 8
    radius = 0.003
    gps_error = 0.0005
    # TODO: has a vmax default of 30 km/h, read in paper how this value affects the results?
    stmatch_config = STMATCHConfig(k, radius, gps_error)

    ### Define GPS configurations
    gps_config = GPSConfig()
    gps_config.file = network_dir + fpoints
    gps_config.id = ID
    gps_config.geom = GEOM
    gps_config.x = X
    gps_config.y = Y
    gps_config.gps_point = GPS_POINT
    
    ### Define result configurations
    result_config = ResultConfig()
    result_config.file = network_dir + MATCHED_OUTPUT
    result_config.output_config.write_opath = True
    
    ### Run map matching for GPS files
    status = model.match_gps_file(gps_config, result_config, stmatch_config, use_omp=True)
    logging.info(status)
 
