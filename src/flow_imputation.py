import copy
import os
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import osmnx as ox
#import sklearn
import logging
#from visualise_network import get_base_graphml
#from osmnx.distance import great_circle
from config import ROOT_DIR
#from collections import defaultdict
from networkx import MultiDiGraph
#from queue import SimpleQueue
from connect_detectors import save_graph_shapefile_directional
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from multiprocessing import Pool, cpu_count


# defines the number k of nearest detectors we look for in
k = 5
networkDataRoot = os.path.join(ROOT_DIR, "data/network")


def find_immediate_neighbors(G: MultiDiGraph) -> dict:
    """
    Store the immediate successor nodes of all network nodes in a dictionary for easier access
    :param G: A MultiDiGraph network
    :return: A dictionary containing lists of the successor nodes of all nodes keyed by the node ids
    """
    nodes = np.array(G.nodes)
    neighbors = G.neighbors(nodes[0])

    dtype = [("id", int), ("length", float)]
    order = "length"

    immediate_neighbors = {}
    for node in nodes:
        neighbors = np.array([n for n in G.neighbors(node)])
        # create a list [id, path_length, path_length]
        neighbors = [(neighbor, G[node][neighbor][0]["length"], G[node][neighbor][0]["length"])
                    for neighbor in neighbors]
        # sort along the last axis which contains the total distance of the neighbor node to the source node 'node'
        sorted_neighbors = sorted(neighbors, key=lambda x: x[-1])
        immediate_neighbors[node] = sorted_neighbors
    return immediate_neighbors


def imputation_scheme(G: MultiDiGraph, nearest_neighbors: dict()) -> MultiDiGraph:
    """
    Imputation scheme according to which the flow value of a node is estimated. The 'use_intermittent_results' flag is
    used to determine whether the estimated flow values should be accessible during the imputation scheme, i.e.
    we can impute the flow while either considering intermittent results, i.e. write imputed flows to the
    graph we read from, or create a copy so we have one 'read' and one 'write' graph

    :param G: A MultiDiGraph network
    :param nearest_neighbors: A dictionary containing the nearest neighbors keyed by the network node ids
    """

    use_intermittent_results = True

    # Create a deepcopy of G to which we write the imputed flow values
    G_write = copy.deepcopy(G)
    # use that copy to read flow values if 'use_intermitten_results' is true, else use the original G
    G_read = G_write if use_intermittent_results else G
    all_flows = nx.get_node_attributes(G_read, 'flow', default=0)

    # used to further scale the imputed flow
    factor = 100
    for node in G_read:
        if G_read.nodes[node]['prior_flow']:
            continue
        if node not in nearest_neighbors:
            continue

        pre = []
        for n in G_read.predecessors(node):
            pre.append(n)

        node_neighbors = nearest_neighbors[node]

        if not node_neighbors:
            continue
       
        # impute the flow by taking the weighted average flow of k detectors using the inverse distances as weight
        flows_distances = np.array(
            [(all_flows[neighbor_id], total_distance) for neighbor_id, total_distance in node_neighbors])
        flows, distances = flows_distances[:, 0], flows_distances[:, 1]
        weights = np.array([factor / (total_distance if total_distance > 0 else 1) for total_distance in distances])
        imputed_flow = np.average(flows, weights=weights)

        G_write.nodes[node]['flow'] = imputed_flow
 
    return G_write


def distance_between_nodes(G: MultiDiGraph, start: int, ends: [int]):
    """
    Calculate the distance between a source node and all detector nodes using the haversine distance
    :param G: The MultiDiGraph network
    :param start: The source node id
    :param ends: A list of detector node ids
    :return: A list of detector node and distance tuples, sorted by distance
    """
    # we need [lat, lon] -> equivalent to [y, x]
    start_coords = [float(G.nodes[start]['y']), float(G.nodes[start]['x'])]
    end_coords = [[float(G.nodes[end]['y']), float(G.nodes[end]['x'])] for end in ends]

    start_coords_radians = [radians(coord) for coord in start_coords]
    end_coords_radians = [[radians(coord) for coord in coords] for coords in end_coords]

    result = haversine_distances([start_coords_radians], end_coords_radians)
    # transform the result to kilometers (earth radius = 6371 km)
    result_scaled = result * 6371000/1000

    distances = list(zip(ends, result_scaled.tolist()[0]))

    return sorted(distances, key=lambda x: x[-1])


def compute_nearest_detectors(args: [int, [int], MultiDiGraph, [int], int]):
    """
    Computes the k-nearest detectors for every node in the network for which we want to impute the flow
    :param args: List containing the node, current_neighbors, G, detectors, k parameters as defined in 'args_list' in
                 the function 'parallelize_knd_computation(G, k, detectors, immediate_neighbors)'
    :return: A tuple of node id and nearest detectors (node, nearest_detectors)
    """
    # unpack argument list
    node, current_neighbors, G, detectors, k = args
    # if the node is a detector with a valid flow, return None, None
    if node in detectors:
        return None, None

    # calculate the distances between the source node and the detectors, result is sorted acc. to distance
    distances = distance_between_nodes(G, node, detectors)
    # take the k^2 nearest detector nodes
    k_sq_distances = distances[:k ** 2]

    nearest_detectors = []
    for detector, _ in k_sq_distances:
        try:
            # calculate the shortest path from node to detector
            dist = nx.astar_path_length(G, node, detector, weight='length')
        except:
            logging.debug("somehow no shortest path was found for node {node} to detector {detector}")
            continue
        # append the (detector, dist) tuple the list of nearest detectors
        nearest_detectors.append((detector, dist))

    # sort the nearest detectors and take the first k entries
    nearest_detectors = sorted(nearest_detectors, key=lambda x: x[-1])[:k]
    return node, nearest_detectors


def parallelize_knd_computation(G, k, detectors, immediate_neighbors):
    """
    Parallelize the k-nearest detectors computation.

    :param G: The MultiDiGraph network
    :param k: The number of nearest detector nodes per source node
    :param detectors: A list containing the detector node ids
    :param immediate_neighbors: A dictionary containing the lists of immediate neighbors keyed by the source node ids
    :return: A dictionary containing the lists of nearest detectors keyed by the source node ids
    """
    knd = {}
    num_processes = cpu_count()
    # create a process pool
    pool = Pool(processes=num_processes)
   
    # define the argument list that is used to map the function execution to the pool of processes
    args_list = [(node, current_neighbors, G, detectors, k) for node, current_neighbors in immediate_neighbors.items()]
   
    # execute 'compute_nearest_detectors in parallel'
    results = pool.map(compute_nearest_detectors, args_list)
    # close the process pool
    pool.close()
    pool.join()

    # store the results in the knd dictionary
    for node, nearest_neighbors in results:
        if node is not None:
            knd[node] = nearest_neighbors

    return knd


def impute():
    """
    Imputation process that estimates the flow value at nodes that have a flow value of 0
    """
    # load the street network of the city
    with open(os.path.join(networkDataRoot, "simplify_nodes_map.gpickle"), 'rb') as f:
        G = pickle.load(f)

    # create attribute that keeps track whether a node contained a flow value prior to imputation
    flow_attr = nx.get_node_attributes(G, 'flow', default=0)
    # INFO: if a detector has 0 flow, its flow value will also be imputed, maybe that is the wrong approach?
    prior_flow = {k: v > 0 for k, v in flow_attr.items()}
    nx.set_node_attributes(G, prior_flow, 'prior_flow')

    # dict of node: (neighbor, distance) pairs
    immediate_neighbors = find_immediate_neighbors(G)

    # create a list of all detector nodes
    detector_nodes = []
    for node, ddict in G.nodes(data=True):
        if ddict['prior_flow'] == True:
            detector_nodes.append(node)

    # execute the parallelized k nearest detectors computation
    nearest_detectors = parallelize_knd_computation(G, k, detector_nodes, immediate_neighbors)

    # debugging
    for node, neighbors in nearest_detectors.items():
        if node in detector_nodes:
            logging.debug(f"source node {node} is in detector_nodes, should not need its flow imputed")
        for neighbor_id, _ in neighbors:
            if neighbor_id not in detector_nodes:
                logging.debug(f"assumed detector neighbor {neighbor_id} is not in detector_nodes")

    # impute the flow values
    G = imputation_scheme(G, nearest_detectors)

    # create a new edge attribute called 'prior_flow', this is used to determine which flow values were imputed
    nx.set_edge_attributes(G, False, 'prior_flow')
    flow_dict = G.nodes.data('flow')
    prior_flow_dict = G.nodes.data('prior_flow')
    # add new edge attributes to G
    for node in G.nodes:
        # should we consider the key value?
        for u, v, ddict in G.edges(node, data=True):
            # if end node of an edge has had no prior flow, it went through imputation -> add flow of v to that edge
            ddict['prior_flow'] = prior_flow_dict[v]
            if not prior_flow_dict[v]:
                # G[u][v]['flow'] = flow_dict[v]
                ddict['flow'] = flow_dict[v]

# TODO: we probably need to save G as a graphml file here again if the imputation happens before the visualisation
    # but i think its visualise -> impute, so the save_graphfile_directional function from visualise_network.py
    # should be moved here, because that function creates a bunch of shp files that we need in add_layer.py
    save_graph_shapefile_directional(G, filepath=networkDataRoot)
    ox.io.save_graph_geopackage(G, os.path.join(networkDataRoot, "detector_nodes.gpkg"))
    with open(os.path.join(networkDataRoot, "imputed_nodes_map.gpickle"), 'wb') as f:
        pickle.dump(G, f)


def main():
    impute()


if __name__ == '__main__':
    main()

