'''
Date: 2024-12-19 13:19:52
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-20 13:22:32
Description: 评估脚本（封装前测试）
'''

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mvn.utils import cfg
from mvn.models.model import LstmModel, TransformerModel
from mvn.datasets.human36m.utils.process import calculate_bone_data
from mvn.utils.rebuild import rebuild_pose_from_root

def get_data_from_csv(data_path):
    """
    从 CSV 文件中读取数据，并将其组织成两个形状为 [frame, 17, 3] 的 NumPy 数组。

    Args:
        data_path: CSV 文件的路径。

    Returns:
        一个元组，包含两个 NumPy 数组：(pred_data, gt_data)，
        其中 pred_data 是预测数据数组，gt_data 是真实值数据数组。
    """

    # 读取 CSV 文件
    df = pd.read_csv(data_path)

    # 提取预测数据列和真实值数据列
    pred_cols = [col for col in df.columns if "_gt" not in col]
    gt_cols = [col for col in df.columns if "_gt" in col]

    # 直接使用 numpy 进行 reshape，提高效率
    pred_data = df[pred_cols].values.reshape(-1, 17, 3) 
    gt_data = df[gt_cols].values.reshape(-1, 17, 3)    

    return pred_data, gt_data

def main(config_path, data_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = cfg.load_config(config_path)

    # model
    model = {
        "lstm": LstmModel,
        "transformer": TransformerModel
    }[config.model.name]

    if config.model.name == "transformer":
        model = model(
            seq_len=config.opt.seq_len,
            num_joints=config.opt.n_joints,
            hidden_size=config.model.n_hidden_layer,
            num_layers=config.model.n_layers,
            num_heads=config.model.n_heads,
            dropout_probability=config.model.dropout
        ).to(device)
    else:
        raise NotImplementedError("Model not implemented")

    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 读取数据
    pred_data, gt_data = get_data_from_csv(data_path)

    pred_data = gt_data

    frames = pred_data.shape[0]
    seq_len = config.opt.seq_len
    valid_frames = frames - seq_len

    pred_data_dict, root_offset = calculate_bone_data(pred_data)

    all_root_outs = []
    all_rotations_outs = []
    all_bone_lengths = []

    model.eval()

    # 初始化前 seq_len 帧
    for i in range(seq_len):
        pred_root = torch.from_numpy(pred_data_dict["root"][i:i + seq_len]).to(device).float().unsqueeze(0)
        pred_rotations = torch.from_numpy(pred_data_dict["rotations"][i:i + seq_len]).to(device).float().unsqueeze(0)
        bone_lengths = torch.from_numpy(pred_data_dict["bone_lengths"][i]).to(device).float().unsqueeze(0)

        root_out, rotations_out = model(pred_root, pred_rotations)

        all_root_outs.append(root_out)
        all_rotations_outs.append(rotations_out)
        all_bone_lengths.append(bone_lengths)

    # 连续畸形数量，防止修正失败
    anomaly_count = 0
    # 预测后续帧
    for i in range(seq_len, valid_frames):
        pred_root = torch.from_numpy(pred_data_dict["root"][i:i + seq_len]).to(device).float().unsqueeze(0)
        pred_rotations = torch.from_numpy(pred_data_dict["rotations"][i:i + seq_len]).to(device).float().unsqueeze(0)
        bone_lengths = torch.from_numpy(pred_data_dict["bone_lengths"][i]).to(device).float().unsqueeze(0)

        root_out, rotations_out = model(pred_root, pred_rotations)

        # 畸形检测
        flag_anomaly = False
        if len(all_root_outs) > 0 and anomaly_count < 3:
            distance = torch.norm(root_out[-1, -2, :] - all_root_outs[-1][-1, -1, :])
            if distance > 220:
                print(f"Detected anomaly at frame {i + seq_len - 1}, using previous frame's result.")
                flag_anomaly = True
        if flag_anomaly:
            pred_root =  all_root_outs[-1][:, 0:, :]
            pred_rotations =  all_rotations_outs[-1][:, 0:, :]
            bone_lengths = torch.from_numpy(pred_data_dict["bone_lengths"][0]).to(device).float().unsqueeze(0)

            root_out, rotations_out = model(pred_root, pred_rotations)
            anomaly_count += 1
        else:
            anomaly_count = 0


        all_root_outs.append(root_out)
        all_rotations_outs.append(rotations_out)
        all_bone_lengths.append(bone_lengths)

    all_root_outs = torch.cat(all_root_outs, dim=0)
    all_root_outs = all_root_outs + torch.from_numpy(root_offset).to(device).float()
    all_rotations_outs = torch.cat(all_rotations_outs, dim=0)
    all_bone_lengths = torch.cat(all_bone_lengths, dim=0)

    joints_gt = torch.from_numpy(gt_data).to(device).float()
    joints_predicted_with_seq = rebuild_pose_from_root(all_root_outs, all_rotations_outs, all_bone_lengths)
    joints_predicted = torch.zeros(frames, 17, 3).to(device).float()
    pred_data = torch.from_numpy(pred_data).to(device).float()
    joints_predicted[0] = pred_data[0]
    joints_predicted[1:seq_len + 1] = joints_predicted_with_seq[0]
    joints_predicted[seq_len + 1:-1] = joints_predicted_with_seq[2:, -2, :, :] 
    joints_predicted[-1] = joints_predicted_with_seq[-1, -1, :, :]


    # 计算每个关节的误差
    error_per_joint = torch.norm(joints_predicted - joints_gt, p=2, dim=-1)

    # 计算所有帧，所有序列和所有关节的平均误差
    mpjpe = error_per_joint.mean()

    print(f"Mean Per Joint Position Error: {mpjpe.item()}")

    joints_predicted = joints_predicted.detach().cpu().numpy()
    visualize_3d_motion(joints_predicted)
def visualize_3d_motion(data, title="3D Motion Visualization"):
    """
    可视化形状为 (frames, 17, 3) 的 3D 动作数据。

    Args:
        data: 形状为 (frames, 17, 3) 的 NumPy 数组或 PyTorch 张量，表示 3D 动作数据。
        title: 图表的标题。
    """

    # 如果是 PyTorch 张量，转换为 NumPy 数组
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # 骨骼连接关系
    skeleton = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
        [0, 7], [7, 8], [8, 9], [9, 10], [8, 14], [14, 15],
        [15, 16], [8, 11], [11, 12], [12, 13]
    ]

    # 创建图形和坐标轴
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴范围 (根据你的数据范围进行调整)
    min_xyz = np.min(data, axis=(0, 1))
    max_xyz = np.max(data, axis=(0, 1))
    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(min_xyz[2], max_xyz[2])

    # 设置标题
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 初始化线条对象 (用于绘制骨骼)
    lines = [ax.plot([], [], [], 'b-')[0] for _ in skeleton]

    # 帧数显示
    frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    # 暂停标志
    is_paused = False

    # 更新函数 (用于动画的每一帧)
    def update(frame):
        if not is_paused:
            # 更新每一根骨骼的位置
            for i, bone in enumerate(skeleton):
                x_data = [data[frame, bone[0], 0], data[frame, bone[1], 0]]
                y_data = [data[frame, bone[0], 1], data[frame, bone[1], 1]]
                z_data = [data[frame, bone[0], 2], data[frame, bone[1], 2]]
                lines[i].set_data(x_data, y_data)
                lines[i].set_3d_properties(z_data)

            # 更新帧数显示
            frame_text.set_text(f"Frame: {frame + 1}/{data.shape[0]}")
            
            # 返回需要更新的对象列表
            return lines + [frame_text]
        else:
            return lines + [frame_text]

    # 键盘事件处理函数
    def on_key(event):
        nonlocal is_paused
        if event.key == ' ':
            is_paused = not is_paused
            if is_paused:
                ani.event_source.stop()
            else:
                ani.event_source.start()

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=33, blit=True)

    # 连接键盘事件
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


if __name__ == '__main__':
    data_path = r"G:\Projects\Joint_track\eval_data\data_no_wrong.csv"

    # config_path = r"G:\Projects\Joint_track\logs\human36m_train_ex_TransformerModel@12-17-23-12-47.2024\config.yaml"
    # model_path = r"G:\Projects\Joint_track\logs\human36m_train_ex_TransformerModel@12-17-23-12-47.2024\checkpoints\1125\weights.pth"

    config_path = r"G:\Projects\Joint_track\logs\human36m_train_ex_TransformerModel@12-19-17-35-22.2024\config.yaml"
    model_path = r"G:\Projects\Joint_track\logs\human36m_train_ex_TransformerModel@12-19-17-35-22.2024\checkpoints\0330\weights.pth"
    main(config_path, data_path, model_path)
    print("Done")
