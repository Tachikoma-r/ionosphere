from dotenv import load_dotenv
import os
import numpy as np
from multiprocessing import freeze_support
from pathlib import Path
from string import ascii_lowercase as alc

from dcb import write_dcb
import logger
from tec import writer

custom_print = logger.get_logger(__name__).info
np.set_printoptions(suppress=True)

pattern = ['22o', '22n']
string_date = '125'
# load_dotenv()

path_to_dcb = 'dcb'
# path_to_dcb=os.getenv('PATH_TO_DCB')
path_to_rx = 'E:/codes/work'
# path_to_rx = 'E:/codes/work1/unzip/5'
# path_to_rx = 'E:/codes/work2'
# path_to_rx=os.getenv('PATH_TO_RINEX')
# path_to_csv = os.getenv('PATH_TO_CSV')
path_to_csv = 'E:/codes/result'

if __name__ == "__main__":
    freeze_support()
    # write_mod = input()
    write_mod = 'dcb'
    match write_mod:
        case 'dcb':
            # write_dcb_new(path_to_csv, path_to_dcb, string_date)#
            write_dcb(path_to_csv, path_to_dcb, string_date, divide=None)
        case 'tec':
            # for d in list(alc[:24]):
            #     writer(path_to_dcb, path_to_csv, path_to_rx, string_date+d, pattern, divide='hour')
            writer(path_to_dcb, path_to_csv, path_to_rx, string_date, pattern)
        case _:
            print('try something')