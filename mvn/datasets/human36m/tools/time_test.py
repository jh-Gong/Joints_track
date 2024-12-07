import h5py
import joblib
import os
import time

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# 测量 HDF5 读取时间
start_time_hdf5 = time.time()

hdf5_data = {}

with h5py.File(os.path.join(current_directory, '..', 'results', 'data_train-origin.h5'), 'r') as h5file:
        for subject in h5file.keys():
            if subject not in hdf5_data:
                hdf5_data[subject] = {}
            subject_group = h5file[subject]

            for action in subject_group.keys():
                if action not in hdf5_data[subject]:
                    hdf5_data[subject][action] = {}
                action_group = subject_group[action]

                for subaction in action_group.keys():
                    if subaction not in hdf5_data[subject][action]:
                        hdf5_data[subject][action][subaction] = {}
                    subaction_group = action_group[subaction]

                    for video_id in subaction_group.keys():
                        if video_id not in hdf5_data[subject][action][subaction]:
                            hdf5_data[subject][action][subaction][video_id] = {}
                        video_group = subaction_group[video_id]

                        joints_3d = video_group['joints_3d'][:]
                        joints_vis = video_group['joints_vis'][:]

                        hdf5_data[subject][action][subaction][video_id]['joints_3d'] = joints_3d
                        hdf5_data[subject][action][subaction][video_id]['joints_vis'] = joints_vis

end_time_hdf5 = time.time()
hdf5_duration = end_time_hdf5 - start_time_hdf5
print(f"HDF5 读取时间: {hdf5_duration:.4f} 秒")

# 测量 joblib 读取时间
start_time_joblib = time.time()

joblib_data = joblib.load(os.path.join(current_directory, '..', 'results', 'data_train-origin.pkl'))

end_time_joblib = time.time()
joblib_duration = end_time_joblib - start_time_joblib
print(f"Joblib 读取时间: {joblib_duration:.4f} 秒")
