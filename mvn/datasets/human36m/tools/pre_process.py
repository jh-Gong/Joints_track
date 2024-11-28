import os
import pickle
import joblib
import numpy as np

def process_data(input_file, output_file, mode):
    try:
        data = joblib.load(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    # 收集所有数据以计算全局统计量
    all_coordinates = []

    for value in data.values():
        all_coordinates.append(np.array(value))

    all_coordinates = np.concatenate(all_coordinates, axis=0)

    scaling_info = {}

    if mode == "norm":
        global_mean = np.mean(all_coordinates, axis=0)
        global_std = np.std(all_coordinates, axis=0)
        global_std[global_std == 0] = 1  # 避免除以零
        scaling_info["mean"] = global_mean.tolist()
        scaling_info["std"] = global_std.tolist()
    
    elif mode == "linear":
        global_min = np.min(all_coordinates, axis=0)
        global_max = np.max(all_coordinates, axis=0)
        global_range = global_max - global_min
        global_range[global_range == 0] = 1  # 避免除以零
        scaling_info["min"] = global_min.tolist()
        scaling_info["max"] = global_max.tolist()
    
    else:
        raise ValueError("Invalid mode. Choose 'norm' or 'linear'.")

    processed_data = {}

    for key, value in data.items():
        coordinates = np.array(value)

        if mode == "norm":
            normalized = (coordinates - global_mean) / global_std
            processed_data[key] = normalized.tolist()
        
        elif mode == "linear":
            scaled = 2 * (coordinates - global_min) / global_range - 1
            processed_data[key] = scaled.tolist()

    # 添加缩放信息到字典中
    processed_data["scaling_info"] = scaling_info

    try:
        joblib.dump(processed_data, output_file)
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")



if __name__ == "__main__":
    mode = "linear"
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    
    input_file_train = os.path.join(current_directory, '..', "results", "data_train-origin.pkl")
    output_file_train = os.path.join(current_directory, '..', "results", f"data_train-{mode}.pkl")
    input_file_val = os.path.join(current_directory, '..', "results", "data_validation-origin.pkl")
    output_file_val = os.path.join(current_directory, '..', "results", f"data_validation-{mode}.pkl")
    
    for input_file, output_file in [(input_file_train, output_file_train), (input_file_val, output_file_val)]:
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
            continue
        
        process_data(input_file, output_file, mode)
