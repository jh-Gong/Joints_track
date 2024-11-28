import os
import matplotlib.pyplot as plt
import numpy as np

def save_3d_png(model, device, dataloader, experiment_dir, name, epoch):
    batch = next(iter(dataloader))
    batch_x, batch_y = batch
    batch_x = batch_x[0]    # size: (seq_len, num_joints * 3)
    batch_y = batch_y[0]    # size: (seq_len + 1, num_joints * 3)
    batch_x = batch_x.unsqueeze(0)
    output = model(batch_x.to(device))  # size: (1, seq_len + 1, num_joints * 3)
    output = output.squeeze(0)

    seq_len_plus_one, num_joints_times_3 = batch_y.shape
    num_joints = num_joints_times_3 // 3

    batch_y = batch_y.view(seq_len_plus_one, num_joints, 3).detach().cpu().numpy()
    output = output.view(seq_len_plus_one, num_joints, 3).detach().cpu().numpy()

    # 关节连接方式
    connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 14), (14, 15),
        (15, 16), (8, 11), (11, 12), (12, 13)
    ]

    save_dir = os.path.join(experiment_dir, 'pose_3d', f"eposh_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    for frame_idx in range(seq_len_plus_one):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=90, azim=90)

        # ground truth 绿色
        for start, end in connections:
            ax.plot(
                [batch_y[frame_idx, start, 0], batch_y[frame_idx, end, 0]],
                [batch_y[frame_idx, start, 1], batch_y[frame_idx, end, 1]],
                [batch_y[frame_idx, start, 2], batch_y[frame_idx, end, 2]],
                'g-'
            )

        # prediction 红色
        for start, end in connections:
            ax.plot(
                [output[frame_idx, start, 0], output[frame_idx, end, 0]],
                [output[frame_idx, start, 1], output[frame_idx, end, 1]],
                [output[frame_idx, start, 2], output[frame_idx, end, 2]],
                'r-'
            )

        all_points = np.concatenate((batch_y[frame_idx], output[frame_idx]), axis=0)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
        # 空余比例
        margin_ratio = 0.2
        x_margin = (x_max - x_min) * margin_ratio
        y_margin = (y_max - y_min) * margin_ratio
        z_margin = (z_max - z_min) * margin_ratio
        ax.set_xlim([x_min - x_margin, x_max + x_margin])
        ax.set_ylim([y_min - y_margin, y_max + y_margin])
        ax.set_zlim([z_min - z_margin, z_max + z_margin])

        ax.set_title(f'Frame {frame_idx}')
        
        # Save the plot
        file_path = os.path.join(save_dir, f'{name}_frame_{frame_idx}.png')
        plt.savefig(file_path)
        plt.close(fig)

        print(f'Saved 3D plot to {file_path}')
