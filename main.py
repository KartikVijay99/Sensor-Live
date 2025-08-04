from sensor.Exception import SensorException
from sensor.Logger import logging
import sys
import os
from sensor.utils import dump_csv_file_to_mongodb_collection






if __name__=="__main__":
    file_path="aps_failure_training_set1.csv"
    database_name="PROJECT"
    collection_name ="Data"
    dump_csv_file_to_mongodb_collection(file_path,database_name,collection_name)    








