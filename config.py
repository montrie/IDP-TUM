import os
import sys

sys.path.insert(2, os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RUNS_PER_RELOAD = 96 * 2 / 96  # number of 15-minute-intervales multiplied with number 
                               # of days -> run the UBODT pre-computation algorithm in match.py provided by FMM
