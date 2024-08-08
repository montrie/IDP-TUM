import copy
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
from queue import SimpleQueue
from visualise_network import save_graph_shapefile_directional
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count


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

    end = time.time_ns()
    print("{}s to run fill_nearest_neighbors with k = {} for {} nodes in G".format((end - start) / 1e9, k, len(immediate_neighbors)))

    return knn


def k_nearest_detectors_astar(G: MultiDiGraph, k: int = 5) -> dict:
    knd = {}
    detectors = []
    for n, ddict in G.nodes(data=True):
        if ddict['prior_flow'] == True: # is the 'True' value in 'prior_flow' saved as a string??
            detectors.append(n)  # TODO: atm, this discards detectors that don't work aka measure 0 flow -> maybe we want them anyway?

    for n in G.nodes:
        # skip node if it is a detector with a flow value
        if n in detectors:
            continue
        sp = []
        print("start astar_path_length timer")
        start = time.time_ns()
        for d in detectors:
            try:
                # TODO: use the correct shortest path function, also parallelize
                sp.append((d, nx.astar_path_length(G, n, d, weight='length')))
            except nx.NetworkXNoPath:
                # ignore detectors that we can't reach
                pass
        end = time.time_ns()
        print("{}s to run astar_path_length for 1 source node to {} detector nodes".format((end - start) / 1e9, len(detectors)))
        print(f"we have {len(G.nodes)} source nodes")
        # TODO: sp should be of form [(node, distande)]
        knd[n] = sorted(sp, key=lambda x: x[-1])[:k]

        break

    return knd


def k_nearest_detectors(G:MultiDiGraph, immediate_neighbors: dict, k: int = 5) -> dict:
    knd = {}
    detectors = []
    for n, ddict in G.nodes(data=True):
        if ddict['prior_flow'] == True: # is the 'True' value in 'prior_flow' saved as a string??
            detectors.append(n)  # TODO: atm, this discards detectors that don't work aka measure 0 flow -> maybe we want them anyway?

    print("start timer")
    start = time.time_ns()

    for node, current_neighbors in immediate_neighbors.items():
    # skip node if it is a detector with a prior flow value
        if node in detectors:
            continue
        q = SimpleQueue()
        _ = [q.put_nowait(item) for item in current_neighbors]
        nearest_neighbors = fill_nearest_detectors(G, node, k, detectors, immediate_neighbors, q, [])
        knn[node] = sorted(nearest_neighbors, key=lambda x: x[-1])

    end = time.time_ns()
    print("{}s to run fill_nearest_detectors with k = {} for {} nodes in G".format((end - start) / 1e9, k, len(immediate_neighbors)))

    return knd


def fill_nearest_detectors(G: MultiDiGraph, base_node: int, k: int, detectors: list, immediate_neighbors: dict, current_neighbors: SimpleQueue, nearest_neighbors: list = [], latest_detectors: list = []):
    # current_neighbors = [(neighbor, distance)]
    # need to keep track of the most recent current_neighbors, because the first few current_neighbors might have closer
    # detectors down the line than the last few current_neighbors
    prev = []
    while not current_neighbors.empty():
        prev.append(current_neighbors.get_nowait())

    # base case: no neighbors -> node is a dead end
    if len(prev) == 0:
        return nearest_neighbors
    # find detector nodes in prev
    nd = [node for node in prev if node in detectors]

    # document what is happening here
    next_neighbors1 = [immediate_neighbors[neighbor[0]] for neighbor in prev]
    next_neighbors = [[n for n in l if n[0] not in list(zip(*nearest_neighbors))[0] and n[0] != base_node] for l in next_neighbors1]

    # when prev has more than one entry, next will be a list of lists -> need to flatten a list that is only potentially a list of lists
    temp = []
    for i, neighbor in enumerate(next_neighbors):
        #TODO: add the distance from the previous entry to the next entry to get the total distance to the base node
        if isinstance(neighbor, list):
            for n in range(len(neighbor)):
                temp.append((next_neighbors[i][n][0], next_neighbors[i][n][1], next_neighbors[i][n][-1] + prev[i][-1]))
        else:
            temp.append((next_neighbors[i][0], next_neighbors[i][1], next_neighbors[i][-1] + prev[i][-1]))

    # some nodes don't have successors -> return current nearest_neighbors
    if len(temp) == 0:
        return nearest_neighbors

    temp = sorted(temp, key=lambda x: x[-1])
    # we need to create successors with the first element of temp included, otw. the next if condition fails 
    # (can't access the first element of an empty list)
    successors = [temp[0]]
    for i in range(1, len(temp)):
        if temp[i][0] not in list(zip(*successors))[0]:
            successors.append(temp[i])

    for neighbor in successors:
        current_neighbors.put_nowait(neighbor)

    # ---------- RECURSIVE CALL ---------- #
    if len(nearest_neighbors) == k:
        # TODO: otw. append nd to newest_detectors, sort, and take the first k detectors, should work because at that point we already keep track of the next nodes we want to visit
        if nd:
            nearest_neighbors = sorted(nearest_neighbors + nd, key=lambda x: x[-1])[:k]
        if successors[0][-1] >= nearest_neighbors[-1][-1]:
        # end recursion if first node of list of nodes were are currently traversing is farther from base_node than the farthest detector in nearest_neighbors
            return nearest_neighbors
        return fill_nearest_detectors(G, base_node, k, detectors, immediate_neighbors, current_neighbors, nearest_neighbors, latest_detectors)
    elif len(nearest_neighbors) + len(nd) >= k:
        nearest_neighbors = sorted(nearest_neighbors + nd, key=lambda x: x[-1])[:k]
        latest_detectors = nd[:rest]
        return fill_nearest_detectors(G, base_node, k, detectors, immediate_neighbors, current_neighbors, nearest_neighbors, latest_detectors)
    else:
        # len(nearest_neighbors) + len(nd) < k
        if nd:
            # we found at least one detector
            nearest_neighbors = sorted(nearest_neighbors + nd, key=lambda x: x[-1])
        return fill_nearest_detectors(G, base_node, k, detectors, immediate_neighbors, current_neighbors, nearest_neighbors)


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
    # test = nearest_neighbors[267723]
    # print(nearest_neighbors[267723])
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
        
        flows_distances = np.array(
            [(all_flows[neighbor_id], total_distance) for neighbor_id, total_distance in node_neighbors])
        flows, distances = flows_distances[:, 0], flows_distances[:, 1]
        weights = np.array([factor / (total_distance if total_distance > 0 else 1) for total_distance in distances])
        imputed_flow = np.average(flows, weights=weights)

#        post = [n for n in node_neighbors if n[1] == n[2]]
#        check = (len(pre) > 1, len(post) > 1)
#
#        if check == (False, False):
#            # only one predecessor and successor
#            # if we have only 1 predecessor and only 1 neighbor where the immediate distance is equal to the
#            # total distance, we have a 'line' of nodes where we can probably just use the same flow value for all nodes on that line
#            # print("check")
#            pre = all_flows[pre[0]] if pre else all_flows[node_neighbors[0][0]]
#            imputed_flow = (pre + all_flows[node_neighbors[0][0]]) / 2
#        #TODO: do we need a case distinction for this or can we just replace this with an else
#        elif check == (False, True) or check == (True, False) or check == (True, True):
#            # (False, False) means multiple predecessors and successors -> we want to probably use all nearest_neighbors
#            flows_distances = np.array(
#                [(all_flows[neighbor_id], total_distance) for neighbor_id, _, total_distance in node_neighbors])
#            # if len(flows_distances.shape) != 2:
#            #     print("check")
#            # try:
#            flows, distances = flows_distances[:, 0], flows_distances[:, 1]
#            # except:
#            #     print("check")
#            # maybe use 100/total_distance instead of 1/total_distance
#            # try:
#            #     for td in distances:
#            #         if td == 0 or td == 0.0:
#            #             print("0 division?")
#            weights = np.array([factor / (total_distance if total_distance > 0 else 1) for total_distance in distances])
#            # except:
#            #     print("Division by 0?")
#            # https://stackoverflow.com/questions/20054243/np-mean-vs-np-average-in-python-numpy
#            imputed_flow = np.average(flows, weights=weights)

        G_write.nodes[node]['flow'] = imputed_flow
 
    return G_write


def distance_between_nodes(G: MultiDiGraph, start: int, ends: [int]):
    # we need [lat, lon] -> equivalent to [y, x]
    start_coords = [float(G.nodes[start]['y']), float(G.nodes[start]['x'])]
    end_coords = [[float(G.nodes[end]['y']), float(G.nodes[end]['x'])] for end in ends]

    start_coords_radians = [radians(coord) for coord in start_coords]
    end_coords_radians = [[radians(coord) for coord in coords] for coords in end_coords]

    result = haversine_distances([start_coords_radians], end_coords_radians)
    result_scaled = result * 6371000/1000  # this gets us the result in kilometers

    distances = list(zip(ends, result_scaled.tolist()[0]))

    return sorted(distances, key=lambda x: x[-1])


def compute_nearest_neighbors(args): #node, current_neighbors, G, detectors, k):
    node, current_neighbors, G, detectors, k = args
    if node in detectors:
        return None, None

    distances = distance_between_nodes(G, node, detectors)
    k_sq_distances = distances[:k * 2]
    nearest_detectors = []

    for detector, _ in k_sq_distances:
        try:
            dist = nx.astar_path_length(G, node, detector, weight='length')
        except:
            continue
        nearest_detectors.append((detector, dist))

    nearest_neighbors = sorted(nearest_detectors, key=lambda x: x[-1])[:k]
    return node, sorted(nearest_neighbors, key=lambda x: x[-1])


def parallelize_neighbors_computation(G, k, detectors, immediate_neighbors):
    knd = {}
    print("start neighbor detectors computation timer using multiprocessing.Pool")
    start = time.time_ns()

#    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
#        futures = {executor.submit(compute_nearest_neighbors, node, current_neighbors, G, detectors, k): node for
#                   node, current_neighbors in immediate_neighbors.items()}
#        for future in as_completed(futures):
#            node, nearest_neighbors = future.result()
#            if node is not None:
#                knd[node] = nearest_neighbors

    num_processes = cpu_count()
    print(f"number of processes used: {num_processes}")
    pool = Pool(processes=num_processes)
    
    args_list = [(node, current_neighbors, G, detectors, k) for node, current_neighbors in immediate_neighbors.items()]
   
    print(f"Time to set up the args_list: {(time.time_ns() - start) / 1e9}")

    results = pool.map(compute_nearest_neighbors, args_list)
    pool.close()
    pool.join()
    
    for node, nearest_neighbors in results:
        if node is not None:
            knd[node] = nearest_neighbors

    end = time.time_ns()
    print("{}s to run parallelize_neighbors_computation with k = {} for {} nodes in G".format((end - start) / 1e9, k,
                                                                                   len(immediate_neighbors)))
    return knd


def impute():
    with open(os.path.join(networkDataRoot, "simplify_nodes_map.gpickle"), 'rb') as f:
        G = pickle.load(f)

    # create attribute that keeps track whether a node contained a flow value prior to imputation
    flow_attr = nx.get_node_attributes(G, 'flow', default=0)
    prior_flow = {k: v > 0 for k, v in flow_attr.items()}
    nx.set_node_attributes(G, prior_flow, 'prior_flow')
    print(f"Number of nodes: {G.number_of_nodes()}")

    # Compute the shortest path distances between nodes
#TODO: parallelize https://stackoverflow.com/questions/69649566/how-do-i-speed-up-all-pairs-dijkstra-path-length

    # dict of node: (neighbor, distance) pairs
    immediate_neighbors = find_immediate_neighbors(G)

    detector_nodes = []
    for n, ddict in G.nodes(data=True):
        if ddict['prior_flow'] == True:  # is the 'True' value in 'prior_flow' saved as a string??
            detector_nodes.append(
                n)  # TODO: atm, this discards detectors that don't work aka measure 0 flow -> maybe we want them anyway?

# TODO: k = min(k, num_detectors)
    k = 5
# TODO: typo
#    nearest_neighbors = k_nearest_neighbors(G, immediate_neighbors, k)
#    nearest_neighbors = k_nearest_detectors_astar(G, k)
#    nearest_neighbors = k_nearest_detectors(G, immediate_neighbors, k)
    nearest_neighbors_parallel = parallelize_neighbors_computation(G, k, detector_nodes, immediate_neighbors)
#    nearest_neighbors_parallel = dict(sorted(nearest_neighbors_parallel.items()))
#    # for k in range(1,30):
#    #     nearest_neighbors = k_nearest_neighbors(G, immediate_neighbors, k)

#    detector_nodes = []
#    for n, ddict in G.nodes(data=True):
#        if ddict['prior_flow'] == True: # is the 'True' value in 'prior_flow' saved as a string??
#            detector_nodes.append(n)  # TODO: atm, this discards detectors that don't work aka measure 0 flow -> maybe we want them anyway?
#    print(detector_nodes)
    for node, neighbors in nearest_neighbors_parallel.items():
        if node in detector_nodes:
            print(f"node {node} is in detector_nodes")
        for neighbor_id, _ in neighbors:
            if neighbor_id not in detector_nodes:
                print(f"assumed detector neighbor {neighbor_id} is not in detector_nodes")

    # impute the flow values according to some scheme
    G = imputation_scheme(G, nearest_neighbors_parallel)

    # add new edge attributes to G
    nx.set_edge_attributes(G, False, 'prior_flow')
# TODO: turn the 'detectors' list into a global var that is None at start of the file and gets filled here
# according to line 213
#TODO: add imputed flow values to the corresponding edges, similar to how its done in visualise_map.py
    flow_dict = G.nodes.data('flow')
    prior_flow_dict = G.nodes.data('prior_flow')
    for node in G.nodes:
#TODO: do i need to consider the key value?
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
    with open(os.path.join(networkDataRoot, "imputed_nodes_map.gpickle"), 'wb') as f:
        # G = pickle.load(f)
        pickle.dump(G, f)


def main():
    impute()


if __name__ == '__main__':
    main()

