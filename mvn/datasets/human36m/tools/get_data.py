import numpy as np
import joblib
import os

def get_data(input_file, output_file):
    try:
        data = joblib.load(input_file)
        print("Load done!")
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    filtered_data = {}
    current_index = 0
    previous_prefix = None

    for item in data:
        original_image_name = item['image'].split('.')[0]
        prefix = '_'.join(original_image_name.split('/')[-1].split('_')[:-1])
        if prefix != previous_prefix:
            current_index += 1
            previous_prefix = prefix
        new_image_name = f"action_{current_index}"

        if new_image_name not in filtered_data:
            filtered_data[new_image_name] = []
        
        filtered_data[new_image_name].append(np.array(item['joints_3d'][:17]))

    joblib.dump(filtered_data, output_file)

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    input_file_train = os.path.join(current_directory, '..', "source", f"h36m_train.pkl")
    output_file_train = os.path.join(current_directory, '..', "results", f"data_train-origin.pkl")
    input_file_val = os.path.join(current_directory, '..', "source", f"h36m_validation.pkl")
    output_file_val = os.path.join(current_directory, '..', "results", f"data_validation-origin.pkl")
    
    for input_file, output_file in [(input_file_train, output_file_train), (input_file_val, output_file_val)]:
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
            continue
        
        get_data(input_file, output_file)
