import copy

import sys
sys.path.insert(-1, r"D:\Program Files\QGIS 3.32.3\apps\gdal\lib\gdalplugins")
sys.path.insert(-1, r"C:\OSGeo4W\apps\gdal\lib\gdalplugins")

import os
import time
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import osmnx as ox
import sklearn
import logging
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from visualise_network import get_base_graphml
from osmnx.distance import great_circle
from config import ROOT_DIR
from collections import defaultdict
from scipy.spatial import KDTree
from networkx import MultiDiGraph
from numba import jit
from queue import SimpleQueue
from src.visualise_network import save_graph_shapefile_directional


networkDataRoot = os.path.join(ROOT_DIR, "data/network")


# TODO: this cannot be sped up with numba, but its fast anyway; 0.1s for 17k nodes
def find_immediate_neighbors(G: MultiDiGraph) -> dict:
    nodes = np.array(G.nodes)
    neighbors = G.neighbors(nodes[0])
    for neighbor in neighbors:
        if G.has_edge(nodes[0], neighbor):
            print("exists")

    dtype = [("id", int), ("length", float)]
    order = "length"
    immediate_neighbors = {}

    print("start timer")
    start = time.time_ns()
    for node in nodes:
        neighbors = np.array([n for n in G.neighbors(node)])
        # neighbors = np.array(
        neighbors = [(neighbor, G[node][neighbor][0]["length"], G[node][neighbor][0]["length"])
                    for neighbor in neighbors]#,
                            # dtype=dtype
                    # )
        # sort along the last axis which contains the
        # sorted_neighbors = np.sort(neighbors, order=order)
        sorted_neighbors = sorted(neighbors, key=lambda x: x[-1])
        # TODO: i dont think we need to check whether we overwrite an existing entry? because node IDs are unique
        immediate_neighbors[node] = sorted_neighbors
        # print(neighbors)
        # break
    end = time.time_ns()
    print("{}s to sort all {} immediate neighbors of each node according to their distance".format((end-start)/1e9, len(immediate_neighbors)))
    logging.info("{}s to sort all {} immediate neighbors of each node according to their distance".format((end-start)/1e9, len(immediate_neighbors)))
    return immediate_neighbors


def fill_nearest_neighbors(G: MultiDiGraph, base_node: int, k: int, immediate_neighbors: dict, current_neighbors: SimpleQueue, nearest_neighbors: list = []):
    # current_neighbors = [(neighbor, distance)]
    # need to keep track of the most recent current_neighbors, because the first few neighbors might have closer
    # neighbors down the line than the last few neighbors
    prev = []
    while not current_neighbors.empty():
        prev.append(current_neighbors.get_nowait())

    # base case:
    if len(nearest_neighbors) + len(prev) == 0:
        return []
    # at least one neighbor exists:
    if len(nearest_neighbors) + len(prev) < k:
        # update nearest_neighbors with neighbors from current_neighbors queue
        nearest_neighbors = sorted(nearest_neighbors + prev, key= lambda x: x[-1])
        # update current_neighbors by getting all neighbors from the old current_neighbors and sorting them by distance
        next1 = [immediate_neighbors[neighbor[0]] for neighbor in prev]
        next = [[n for n in l if n[0] not in list(zip(*nearest_neighbors))[0] and n[0] != base_node] for l in next1]
        # next = [n for neighbor in prev for n in immediate_neighbors[neighbor[0]] if n[0] not in list(zip(*nearest_neighbors))[0]]
        # if [] in next:
        #     print("check")

        # when prev has more than one entry, next will be a list of lists -> need to flatten a list that is only potentially a list of lists
        temp = []
        for i, neighbor in enumerate(next):
            #TODO: add the distance from the previous entry to the next entry to get the total distance to the base node
            # next[i][-1] += prev[i][-1]
            if isinstance(neighbor, list):
                for n in range(len(neighbor)):
                    # temp.append(next[i][n:] + [(next[i][n][0], next[i][n][1], next[i][n][-1] + prev[i][-1])] + next[i][n + 1:])
                    temp.append((next[i][n][0], next[i][n][1], next[i][n][-1] + prev[i][-1]))
            else:
                # next = next[:i] + [(next[i][0], next[i][1], next[i][-1] + prev[i][-1])] + next[i+1:]
                temp.append((next[i][0], next[i][1], next[i][-1] + prev[i][-1]))

        # some nodes don't have successors
        if len(temp) == 0:
            return nearest_neighbors

        # there are cases, e.g. when base_node == 130010, where successors will look like this
        # [(130020, 102.14500000000001, 243.95100000000002), (365445, 98.244, 267.11300000000006),
        #  (130007, 79.058, 276.257), (60876692, 145.89300000000003, 287.69900000000007), (60876692, 33.657, 289.958),
        #  (130025, 29.358, 352.842), (130004, 199.19099999999997, 396.39), (564038, 270.877, 594.361)]
        # how to handle the case of id 60876692? just take the first occurence (shortest_path to base_node) and throw away the rest?
        temp = sorted(temp, key=lambda x: x[-1])
        successors = [temp[0]]
        for i in range(1, len(temp)):
            if temp[i][0] not in list(zip(*successors))[0]:
                successors.append(temp[i])

        for neighbor in successors:
            current_neighbors.put_nowait(neighbor)
        # recursive call
        return fill_nearest_neighbors(G, base_node, k, immediate_neighbors, current_neighbors, nearest_neighbors)
    else:
        rest = k - len(nearest_neighbors)
        # newest = [current_neighbors.get_nowait() for _ in range(rest)]  #[:rest]
        newest = prev[:rest]
        changed = True
        # check for all but the last newest neighbor whether it has a successor that is closer to the base node than
        # the next entry in newest -> replace the next entry in newest with
        while changed:
            changed = False
            # placeholder list for potential neighbors that are closer to base_node but have more intermittent nodes on the path
            closer = newest
            i = 0
            # the last element of newest cannot possibly have neighbors that are closer than itself
            for i in range(len(newest) - 1): # or reset changed here??
                candidates = immediate_neighbors[newest[i][0]]
                for candidate in candidates:
                    # TODO: probably need to do closer = newest here already? or maybe not?? maybe we can reset 'changed' here too?
                    # skip candidate if it is already part of newest or nearest_neighbor, since that is a neighbor we already reached once
                    if candidate[0] in list(zip(*(nearest_neighbors + closer)))[0]:
                        continue
                    for j, neighbor in enumerate(newest[i+1:]):
                        # do not consider neighbors that we already visited
                        if neighbor[0] in list(zip(*(nearest_neighbors + closer)))[0]:
                            continue
                        # check if the distance from base_node to candidate is less than to neighbor -> TODO: do I need to use i or j here?
                        if closer[i][-1] + candidate[-1] < neighbor[-1]:
                            # create a new list which take the first i+1 elements from closer, adds the closer candidate neighbor
                            # and drops the last (i.e. furthest) neighbor in newest
                            # TODO: ugly, what if the 'neighbor' tuples change layout again?
                            # candidate = *candidate[:-1], candidate[-1] + closer[i][-1]
                            candidate = candidate[0], candidate[1], candidate[-1] + closer[i][-1]
                            # since distances are always >= 0, closer is already sorted, and we know here that
                            # closer[i][-1] + candidate[-1] < neighbor[-1] (= newest[i+1]), we can insert candidate at i+1
                            closer = closer[:j+1] + [candidate] + closer[j+1:-1]
                            changed = True
                    newest = sorted(closer, key= lambda x: x[-1])
                    if not changed:
                        break
        # forgot the return statement :)
        return nearest_neighbors + newest


# def k_nearest_neighbors(G: MultiDiGraph, neighbors: {int: (int, float)}, k: int = 5) -> dict:
def k_nearest_neighbors(G: MultiDiGraph, immediate_neighbors: dict, k: int = 5) -> dict:
    # implement a BFS that finds the k nearest neighbors of each node of G
    # neighbors contains k,v pairs where k = node_id and v = (neighbor, distance)
    # neighbors is sorted according to the distance to the key node
    knn = {}
    print("start timer")
    start = time.time_ns()

    for i, (node, current_neighbors) in enumerate(immediate_neighbors.items()):
        # Put elements from current_neighbors into FIFO queue q
        q = SimpleQueue()
        _ = [q.put_nowait(item) for item in current_neighbors]
        nearest_neighbors = fill_nearest_neighbors(G, node, k, immediate_neighbors, q)
        knn[node] = sorted(nearest_neighbors, key=lambda x: x[-1])
        # print(len(knn))
        # break

    end = time.time_ns()
    print("{}s to run fill_nearest_neighbors with k = {} for {} nodes in G".format((end - start) / 1e9, k, len(immediate_neighbors)))

    return knn


def imputation_scheme(G: MultiDiGraph,nearest_neighbors: dict()) -> MultiDiGraph:
    # TODO: we can impute the flow while either considering intermittent results, i.e. write imputed flows to the
    # graph we read from, or create a copy so we have one 'read' and one 'write' graph
    # maybe we can somehow determine a good place to start the imputation? instead of simply iterating over the nodes
    # in the order they sorted in G

    use_intermittent_results = True

    # Create a deepcopy of G to which we write the imputed flow values
    G_write = copy.deepcopy(G)
    # use that copy to read flow values if 'use_intermitten_results' is true, else use the original G
    G_read = G_write if use_intermittent_results else G
    all_flows = nx.get_node_attributes(G_read, 'flow', default=0)

    factor = 100
    for node in G_read:
        if G_read.nodes[node]['prior_flow']:
            continue
        pre = []
        for n in G_read.predecessors(node):
            pre.append(n)
        if len(pre) == 1:
            # if we have only 1 predecessor and only 1 neighbor where the immediate distance is equal to the
            # total distance, we have a 'line' of nodes where we can probably just use the same flow value for all nodes on that line
            print("check")
        node_neighbors = nearest_neighbors[node]
        flows_distances = np.array([(all_flows[neighbor_id], total_distance) for neighbor_id, _, total_distance in node_neighbors])
        flows, distances = flows_distances[:, 0], flows_distances[:, 1]
        # maybe use 100/total_distance instead of 1/total_distance
        weights = np.array([factor/total_distance for total_distance in distances])
        # https://stackoverflow.com/questions/20054243/np-mean-vs-np-average-in-python-numpy
        imputed_flow = np.average(flows, weights=weights)
        G_write.nodes[node]['flow'] = imputed_flow
 
    return G_write


def impute():
    with open(os.path.join(networkDataRoot, "simplify_nodes_map.gpickle"), 'rb') as f:
        G = pickle.load(f)
    print(f"sklearn version: {sklearn.__version__}")

#     # Extract and print all unique edge attribute names
#     edge_attrs = set()
#     for u, v, key, attr in G.edges(keys=True, data=True):
#         edge_attrs.update(attr.keys())
#     print("Edge attribute names:", edge_attrs)
#
#     # extract edges and attributes into a DataFrame
#     edges = []
#     for u, v, key, data in G.edges(keys=True, data=True):
#         edges.append((u, v, key, data.get('flow', 0), data))
#     df = pd.DataFrame(edges, columns=['u', 'v', 'key', 'flow', 'attributes'])
#
#     # add specific features
#     df['length'] = df['attributes'].apply(lambda x: x.get('length', 0))
# #    df['speed_limit'] = df['attributes'].apply(lambda x: x.get('maxspeed', 0))
#     df_knn = df.drop(columns=["attributes", "key", "flow"])
#     print(df_knn.head())

    # create attribute that keeps track whether a node contained a flow value prior to imputation
    flow_attr = nx.get_node_attributes(G, 'flow', default=0)
    prior_flow = {k: v > 0 for k, v in flow_attr.items()}
    nx.set_node_attributes(G, prior_flow, 'prior_flow')

    # # Extract and print all unique node attribute names
    # node_attrs = set()
    # for node, attr in G.nodes(data=True):
    #     node_attrs.update(attr.keys())
    # print("Node attribute names:", node_attrs)

    # Compute the shortest path distances between nodes
#TODO: parallelize https://stackoverflow.com/questions/69649566/how-do-i-speed-up-all-pairs-dijkstra-path-length
    print(f"Number of nodes: {G.number_of_nodes()}")

    # dict of node: (neighbor, distance) pairs
    immediate_neighbors = find_immediate_neighbors(G)
    k = 7
    nearest_neighors = k_nearest_neighbors(G, immediate_neighbors, k)
    # for k in range(1,30):
    #     nearest_neighors = k_nearest_neighbors(G, immediate_neighbors, k)
    print("check")

    # impute the flow values according to some scheme
    G = imputation_scheme(G, nearest_neighors)

    # TODO: we probably need to save G as a graphml file here again if the imputation happens before the visualisation
    # but i think its visualise -> impute, so the save_graphfile_directional function from visualise_network.py
    # should be moved here, because that function creates a bunch of shp files that we need in add_layer.py
    save_graph_shapefile_directional(G)
    with open(os.path.join(networkDataRoot, "imputed_nodes_map.gpickle"), 'wb') as f:
        # G = pickle.load(f)
        pickle.dump(G, f)


def main():
    impute()


if __name__ == '__main__':
    main()