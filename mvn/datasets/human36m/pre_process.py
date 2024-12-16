r'''
Author: gjhhh 1377019164@qq.com
Date: 2024-11-28 14:11:09
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-12 20:45:20
FilePath: \Joint_track\mvn\datasets\human36m\pre_process.py
Description: 预处理数据集入口文件
'''
import os

from utils.process import process_data


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    
    input_file_train = os.path.join(current_directory, "results", "data_train-origin.h5")
    output_file_train = os.path.join(current_directory, "results", "data_train-latest.h5")

    input_file_val = os.path.join(current_directory, "results", "data_validation-origin.h5")
    output_file_val = os.path.join(current_directory, "results", "data_validation-latest.h5")
    
    for input_file, output_file in [(input_file_train, output_file_train), (input_file_val, output_file_val)]:
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
            continue
        
        process_data(input_file, output_file)
