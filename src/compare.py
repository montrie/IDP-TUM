import asyncio
import csv
import pandas as pd
import geopandas as gpd
import time
import datetime
import glob
import os
import logging

from config import ROOT_DIR


outputDataRoot = os.path.join(ROOT_DIR, "data/output_data/")
comparedDataRoot = os.path.join(ROOT_DIR, "data/compared_data/")
mergedDataRoot = os.path.join(ROOT_DIR, "data/merged_data/")
staticDataRoot = os.path.join(ROOT_DIR, "data/static_data/")
xmlDataRoot = os.path.join(ROOT_DIR, "data/xml_data/")

lrz_munich_loops_path = os.path.join(ROOT_DIR, "data/LRZ/munich_loops_mapping_in_progress.csv")
lrz_csv_file_path = os.path.join(ROOT_DIR, "data/LRZ/munich_loops_mapping_in_progress.csv")
lrz_shp_file_path = os.path.join(ROOT_DIR, "data/LRZ/munich_loops_mapping_in_progress.shp")

# read LRZ shapefile and convert to csv file
df = gpd.read_file(lrz_shp_file_path)
# rename x/y columns to lon/lat
df['lat'] = df['geometry'].y
df['lon'] = df['geometry'].x
df.rename(columns={'DETEKTO':'detid'}, inplace=True)
df.to_csv(lrz_csv_file_path, sep=',', encoding='utf-8')


def compare_csv_files():
    """
    Incoorporate the hand-labelled detector location into the static data
    """
    latest_output_data_csv = max(glob.glob(outputDataRoot+"*.csv"), key=os.path.getctime)

    hand_labelled_locations = pd.read_csv(lrz_munich_loops_path, index_col=0)
    output_data = pd.read_csv(latest_output_data_csv, index_col=0)

    # a LEFT JOIN b on lat, lon columns
    output = hand_labelled_locations.merge(output_data, on=["lat", "lon"], how="left").fillna(0).set_index("detid")
    output.to_csv(comparedDataRoot+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace("-", "_")+"_output.csv")

    latest_compared_data_csv = max(glob.glob(comparedDataRoot+"*.csv"), key=os.path.getctime)
    df = pd.read_csv(latest_compared_data_csv)
    del df['geometry']
    df.to_csv(mergedDataRoot+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace("-", "_")+"_merged.csv", sep=',', encoding='utf-8')

    logging.info("Finished merging measurement data")


def output_data():
    """
    Merge static location data and traffic properties. Remove entries from the latest detector data containing
    empty entries (i.e. 0) in the lat/lon fields.
    """

    latest_xml_data_csv = max(glob.glob(xmlDataRoot + "*.csv"), key=os.path.getctime)
    latest_static_data_csv = max(glob.glob(staticDataRoot + "*static.csv"), key=os.path.getctime)

    # merge_detector_locations(latest_static_data_csv)

    xml_data = pd.read_csv(latest_xml_data_csv, index_col=0)
    static_data = pd.read_csv(latest_static_data_csv, index_col=0)

    output = xml_data.merge(static_data, on="detid", how="left").fillna(0).set_index("detid")
    output.to_csv(mergedDataRoot + "test_output_pre_clear.csv")
#    zero_coords_filter = (output['lat'] == 0) | (output['lat'] == 0.0) | (output['lon'] == 0) | (output['lon'] == 0.0)
#    output = output[~zero_coords_filter]

    output.to_csv(
        os.path.join(
            outputDataRoot,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace("-", "_") + "_output.csv"
        )
    )



if __name__ == "__main__":
    output_data()
    compare_csv_files()

