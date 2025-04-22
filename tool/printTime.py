import numpy as np
import datetime

def printCurrentDatetime():
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S: ')
    return current_datetime