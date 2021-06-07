#import tensorflow as tf
from Settings import settings as s
from Data_Handler import Load_Data_Frames as l
import os
import glob
from pathlib import Path
from Template_Matching import TemplateMatching as T
import cv2
import pandas as pd
if __name__ == '__main__':
    print('Hello')
    # #print ("Num GPUs Available: ", len (tf.config.list_physical_devices ('GPU')))
    #l.path_update('Data/patients_data.csv')
    # from Data_Handler import Load_Data_Frames as L
    # df = pd.read_csv('a.csv')
    #
    # p = L.DataInfo(df, "aaa")
    # p.prepare_dicts()
    # p.data_checking()
    # p.stats()
    # p.plot()
    #T.preprocess_images()