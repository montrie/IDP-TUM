from fmm import FastMapMatch, FastMapMatchConfig, Network, NetworkGraph, UBODTGenAlgorithm, UBODT 
from fmm import GPSConfig, ResultConfig
from config import RUNS_PER_RELOAD

import sys
import logging

### Get cl args
#argscount = 2
#if len(sys.argv) != argscount+1:
#    print(("We need {} args, namely: {}, {}".format(argscount, "network_dir", "points")))
#    quit()
#network_dir, fpoints = sys.argv[1:]
#with open(fpoints, "r") as f:
#    wkt = f.read()
#    print(wkt)

network_dir = "../data/network"
runs = 0


def match(fpoints: str):
    # Declare runs as global so we can modify it
    global runs

    ### Read network data
    network = Network(network_dir+"/edges.shp","fid","u","v")
    print("Nodes {} edges {}".format(network.get_node_count(),network.get_edge_count()))
    graph = NetworkGraph(network)
       
    ### Precompute an UBODT table  
    # Can be skipped if you already generated an ubodt file
    runs %= RUNS_PER_RELOAD
    if (runs == 0):
        ubodt_gen = UBODTGenAlgorithm(network,graph)
        status = ubodt_gen.generate_ubodt(network_dir+"/ubodt.txt", 0.02, binary=False, use_omp=True)
        print(status)
        logging.info(f"Precomputed the UBODT table in {__name__}")
    else:
        logging.info(f"Did not precompute the UBODT table in {__name__}")
    runs += 1
    
    ### Read UBODT
    ubodt = UBODT.read_ubodt_csv(network_dir+"/ubodt.txt")
    
    ### Create FMM model
    model = FastMapMatch(network,graph,ubodt)
    
    ### Define map matching configurations
    k = 8
    radius = 0.003
    gps_error = 0.0005
    fmm_config = FastMapMatchConfig(k,radius,gps_error)

    ### Define GPS configurations
    gps_config = GPSConfig()
#                           file_arg=fpoints,
#                           id_arg='detid',
#                           geom_arg='geometry',
#                           x_arg='lon',
#                           y_arg='lat',
#                           gps_point_arg=True)
    gps_config.file = network_dir + "/" + fpoints
    gps_config.id = 'detid'
    gps_config.geom = 'geometry'
    gps_config.x = 'lon'
    gps_config.y = 'lat'
    gps_config.gps_point = True
    
    ### Define result configurations
    result_config = ResultConfig()
    result_config.file = network_dir + "/matched.csv"  # + fpoints[:-4] + "matched.csv"  # fpoints[:-4] strips the file ending
    result_config.output_config.write_opath = True
    
    ### Run map matching for GPS files
    status = model.match_gps_file(gps_config, result_config, fmm_config)
    print(status)
    
    ### Run map matching for wkt
    # wkt = "LineString(104.10348 30.71363,104.10348 30.71363,104.10348 30.71363,104.10348 30.71363,104.10348 30.71363)"
    # result = model.match_wkt(wkt,fmm_config) # wkt comes from cl input
    
    ### Print map matching result
    # print("Opath ",list(result.opath))
    # print("Cpath ",list(result.cpath))
    # print("WKT ",result.mgeom.export_wkt())
    
    # TODO: 
    """
    1. find out how to modify FastMapMatchConfig to mimic the behaviour of the command line execution of fmm:
       fmm --ubodt "ubodt.txt" --network "edges.shp" --network_id "fid" --source "u" --target "v" --gps "points.csv" --gps_id "detid" --gps_x 'lon' --gps_y 'lat' --gps_geom "geometry" --gps_point 1 --output "matched.csv" -k 8 -r 0.003 -e 0.0005
    2. either use the script above or write a CL script that is then executed by python mimicking command line behaviour
    
    -> there is a bunch of config classes, can probably use these to mimick behaviour of command line call of fmm
    """
