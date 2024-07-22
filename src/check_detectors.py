import glob
import os
import pickle
import pandas as pd
import numpy as np
from config import ROOT_DIR
from pull_xml_to_csv_mobilithek import xml_to_csv
from pull_static_mobilithek import static_data

CLEAR = False

staticDataRoot = os.path.join(ROOT_DIR, "data/static_data")
xmlDataRoot = os.path.join(ROOT_DIR, "data/xml_data")

newest_static_csv = max(glob.glob(os.path.join(staticDataRoot, "*.csv")), key=os.path.getctime)
newest_xml_csv = max(glob.glob(os.path.join(xmlDataRoot, "*.csv")), key=os.path.getctime)
static_dict_path = os.path.join(staticDataRoot, "unknown_location_count_dict.pickle")
xml_dict_path = os.path.join(xmlDataRoot, "unknown_flow_count_dict.pickle")
static_csv_path = os.path.join(staticDataRoot, "unknown_location_count_dict.csv")
xml_csv_path = os.path.join(xmlDataRoot, "unknown_flow_count_dict.csv")

dict_paths = [static_dict_path, xml_dict_path]
csv_paths = [static_csv_path, xml_csv_path]


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    obj = dict()
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj


def clear_count_dicts(paths:[str] = dict_paths):
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)


def to_csv(paths:[(str, str)] = zip(dict_paths, csv_paths)):
    for dict_path, csv_path in paths:
        d = load_pickle(dict_path)
        if len(d) == 0:
            continue
        df = pd.DataFrame.from_dict(d, orient='index')
        df.to_csv(csv_path)


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
    if CLEAR:
        clear_count_dicts()
    check_static_csv()
    check_xml_csv()
    to_csv()


def main():
    static_data()
    xml_to_csv()
    check_detectors()


if __name__ == '__main__':
    main()

