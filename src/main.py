from datetime import datetime
from add_layer import layer
from pull_xml_to_csv_mobilithek import xml_to_csv
from pull_static_mobilithek import static_data
from compare import output_data, compare_csv_files
#from clear_flow import clear_flow
from flow_imputation import impute
from connect_detectors import split_and_reconnect
from website import update_website
from config import ROOT_DIR

import logging


def workflow():
    """
    Main workflow that downloads the static and measurement data, 
    """
    # download and store the measurement data
    xml_to_csv() 
    # download and store the static data
    static_data()
    # merge static location data and measurement data into one file
    output_data()
    # use hand-labelled detector locations to complement the detector coordinates
    compare_csv_files()
#    clear_flow()
    # splice and reconnect the network edges
    split_and_reconnect()
    # impute detector flow values for every network node
    impute()
    # create the qgis layer containing the visualization
    layer()
    # push files that update the website
    update_website() 


def setup_logging():
    """
    Setup the basic configuration of the logging module
    """
    logging.basicConfig(level=logging.INFO, filename="log.log", filemode="w",
                        format="%(name)s %(asctime)s %(levelname)s %(message)s")


def main():
    """
    Entry point for the processing pipeline. Sets up the logging module
    and executes the workflow
    """
    setup_logging()
    workflow()
    logging.info("-----END OF WORKFLOW-----")


if __name__ == '__main__':
       main()
