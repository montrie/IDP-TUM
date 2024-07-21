import glob
import os
import pickle
import pandas as pd
import numpy as np
from config import ROOT_DIR

staticDataRoot = os.path.join(ROOT_DIR, "data/static_data")
xmlDataRoot = os.path.join(ROOT_DIR, "data/xml_data")

newest_static_csv = max(glob.glob(os.path.join(staticDataRoot, "*.csv")), key=os.path.getctime)
newest_xml_csv = max(glob.glob(os.path.join(xmlDataRoot, "*.csv")), key=os.path.getctime)
static_dict_path = os.path.join(staticDataRoot, "unknown_location_count_dict.pickle")
xml_dict_path = os.path.join(xmlDataRoot, "unknown_flow_count_dict.pickle")

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        # pass

def load_pickle(path):
    obj = dict()
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj


def check_static_csv(csv_path = newest_static_csv):
    oldd = load_pickle(static_dict_path)
    csv_df = pd.read_csv(csv_path, index_col=0)
    newd = dict()
    for id, row in csv_df.iterrows():
        if row.isna()['lon'] or row.isna()['lat']:
            newd[row['detid']] = 1

    for entry in newd:
        oldd[entry] = oldd.get(entry, 0) + newd[entry]
    save_pickle(oldd, static_dict_path)


def check_xml_csv(csv_path = newest_xml_csv):
    oldd = load_pickle(xml_dict_path)
    csv_df = pd.read_csv(csv_path, index_col=0)
    newd = dict()
    for id, row in csv_df.iterrows():
        if row['flow'] == 0:
            oldd[row['detid']] = 1

    for entry in newd:
        oldd[entry] = oldd.get(entry, 0) + newd[entry]
    save_pickle(oldd, xml_dict_path)


def check_detectors():
    check_static_csv()
    check_xml_csv()


def main():
    check_detectors()


if __name__ == '__main__':
    main()

