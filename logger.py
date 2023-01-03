import datetime
import logging
from pathlib import Path
from PyQt5.QtWidgets import QTextEdit


def get_logger(name):
    base_dir = '../'
    # Make Dir If Not Exists
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    _log_file = base_dir + "codes/tmp/logfile.log"

    file_handler = logging.FileHandler(_log_file)
    file_handler.setLevel(logging.INFO)

    file_log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(" \
                      f"threadName)s - %(message)s "
    file_handler.setFormatter(logging.Formatter(file_log_format))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    str_log_format = "\x1b[0;32m%(asctime)s \x1b[1;31m %(name)s \x1b[0;33m(%(filename)s).%(funcName)s(%(lineno)d) " \
                     "\x1b[0;34m%(threadName)s\n" + "\x1b[0;37m%(message)s\x1b[0;37m"  # powershell coloring
    stream_handler.setFormatter(logging.Formatter(str_log_format))
    logger.addHandler(stream_handler)

    return logger


class Log_ui:

    log_field: QTextEdit

    def __int__(self: QTextEdit):
        Log_ui.log_field = self

    @staticmethod
    def add_log(log: str):
        Log_ui.log_field.append(f"{datetime.datetime.now()} - [INFO] - {log}.")

