from sensor.Exception import SensorException
from sensor.Logger import logging
import sys
import os



if __name__=="__main__":

    try:
        raise SensorException("this is a test exception", sys)
    except Exception as e:
        raise SensorException(str(e), sys)    









