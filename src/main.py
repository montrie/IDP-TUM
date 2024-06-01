from add_layer import layer
from datetime import datetime
from time import sleep
from pull_xml_to_csv_mobilithek import xml_to_csv
from pull_static_mobilithek import static_data
from compare import output_data, compare_csv_files
from clear_flow import clear_flow
from visualise_network import plot
from website import update_website
from config import ROOT_DIR

import cProfile
import logging
import warnings

LOOP = False
PROFILE = False  # set to True to include profiling
INTERVAL = 15  # defines at which multiple of 60 minutes in an hour we want to run task()
# warnings.simplefilter('error', DeprecationWarning)


def task():
        xml_to_csv()
        static_data()
        output_data()
        compare_csv_files()
        clear_flow()
        plot() 
        layer()
        update_website()


def run(condition: bool):
    """ For now: Entry point for the processing pipeline. Calls the different processing
    steps before sleeping for 15 minutes if ``condition`` is ``True``

    :param condition: specifies whether processing pipeline should run indefinitely
    """
#    while datetime.now().minute not in range(0, 60, INTERVAL):
#        # Wait 1 second until we are synced up with the 'every 15 minutes' clock
#        sleep(60)

    task()

#    if condition:
#        run(condition)


def setup_logging():
    """
    Setup the basic configuration of the logging module
    """
    logging.basicConfig(level=logging.INFO, filename="log.log", filemode="w",
                        format="%(name)s %(asctime)s %(levelname)s %(message)s")


def profile():
    # TODO: figure out how to profile the code, maybe we can fix the slow parts
    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    pr.print_stats(sort="time")
    pr.dump_stats(ROOT_DIR + "/profiling/profile.prof")


def main():
    xml_to_csv()
    static_data()
    output_data()
    compare_csv_files()
    clear_flow()
    plot()
    layer()


if __name__ == '__main__':
    setup_logging()
    if PROFILE:
        profile()
    else:
       # main()
       run(LOOP)
