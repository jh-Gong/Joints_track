r'''
Author: gjhhh 1377019164@qq.com
Date: 2024-11-27 16:23:02
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-12 20:08:36
FilePath: \Joint_track\mvn\datasets\human36m\tools\get_data.py
Description: 获取origin数据以预处理
'''

import os

from utils.formats import get_data_from_csv_to_hdf5

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)

    input_file_train = os.path.join(current_directory, "source", "train_csv")
    output_file_train = os.path.join(current_directory, "results", f"data_train-origin.h5")

    input_file_val = os.path.join(current_directory, "source", "val_csv")
    output_file_val = os.path.join(current_directory, "results", f"data_validation-origin.h5")
    
    get_data_from_csv_to_hdf5(input_file_train, output_file_train, True)
    get_data_from_csv_to_hdf5(input_file_val, output_file_val, True)