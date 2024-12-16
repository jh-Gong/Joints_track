import os
import numpy as np
import torch
from tqdm import tqdm
import plotly.graph_objects as go

from mvn.utils.rebuild import rebuild_pose_from_root

def save_3d_png(model, device, dataloader, experiment_dir, name, epoch):
    model.eval()
    batch = next(iter(dataloader))
    batch_root_x = batch[0]['root'][0:1].to(device)  # (1, seq_len, 3)
    batch_rotations_x = batch[0]['rotations'][0:1].to(device) # (1, seq_len, 16 * 4)
    batch_bone_lengths = batch[0]['bone_lengths'][0:1].to(device) # (1, num_joints - 1)

    batch_root_y = batch[1]['root'][0:1].to(device)
    batch_rotations_y = batch[1]['rotations'][0:1].to(device)

    root_out, rotations_out = model(batch_root_x, batch_rotations_x)

    seq_len = root_out.shape[1]

    # 计算预测三维坐标
    output = rebuild_pose_from_root(root_out, rotations_out, batch_bone_lengths)[0].detach().cpu().numpy()

    # 计算gt三维坐标
    batch_y = rebuild_pose_from_root(batch_root_y, batch_rotations_y, batch_bone_lengths)[0].detach().cpu().numpy()

    # # 关节连接方式
    connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 14), (14, 15),
        (15, 16), (8, 11), (11, 12), (12, 13)
    ]

    save_dir = os.path.join(experiment_dir, 'pose_3d', f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    file_list = []

    # 遍历每一帧
    for frame_idx in range(seq_len):

        # 创建散点图数据
        scatter_gt = go.Scatter3d(
            x=batch_y[frame_idx, :, 0],
            y=batch_y[frame_idx, :, 1],
            z=batch_y[frame_idx, :, 2],
            mode='markers',
            marker=dict(size=5, color='green'),
            name='Ground Truth'
        )

        scatter_pred = go.Scatter3d(
            x=output[frame_idx, :, 0],
            y=output[frame_idx, :, 1],
            z=output[frame_idx, :, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Prediction'
        )

        # 创建线段图数据（用于连接关节）
        lines_gt = []
        lines_pred = []
        # 用于连线的标签列表
        lines_labels_gt = []
        lines_labels_pred = []
        for idx, (start, end) in enumerate(connections):
            lines_gt.append(go.Scatter3d(
                x=[batch_y[frame_idx, start, 0], batch_y[frame_idx, end, 0]],
                y=[batch_y[frame_idx, start, 1], batch_y[frame_idx, end, 1]],
                z=[batch_y[frame_idx, start, 2], batch_y[frame_idx, end, 2]],
                mode='lines+text',
                line=dict(color='green', width=2),
                text=[f'{start}', f'{end}'],
                textposition="top center",
                showlegend=False,
                textfont=dict(color='green')
            ))
            lines_labels_gt.append(f'{start}-{end}') # 添加连线标签

            lines_pred.append(go.Scatter3d(
                x=[output[frame_idx, start, 0], output[frame_idx, end, 0]],
                y=[output[frame_idx, start, 1], output[frame_idx, end, 1]],
                z=[output[frame_idx, start, 2], output[frame_idx, end, 2]],
                mode='lines+text',
                line=dict(color='red', width=2),
                text=[f'{start}', f'{end}'],
                textposition="top center",
                showlegend=False,
                textfont=dict(color='red')
            ))
            lines_labels_pred.append(f'{start}-{end}')

        # 合并数据
        data = [scatter_gt, scatter_pred] + lines_gt + lines_pred

        # 设置布局
        all_points = np.concatenate((batch_y[frame_idx], output[frame_idx]), axis=0)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
        margin_ratio = 0.2
        x_margin = (x_max - x_min) * margin_ratio
        y_margin = (y_max - y_min) * margin_ratio
        z_margin = (z_max - z_min) * margin_ratio

        layout = go.Layout(
            title=f'Frame {frame_idx}',
            scene=dict(
                xaxis=dict(range=[x_min - x_margin, x_max + x_margin]),
                yaxis=dict(range=[y_min - y_margin, y_max + y_margin]),
                zaxis=dict(range=[z_min - z_margin, z_max + z_margin]),
                aspectmode='cube'  # 保持 xyz 轴比例一致
            )
        )

        # 创建图形并保存为 HTML
        fig = go.Figure(data=data, layout=layout)
        file_path = os.path.join(save_dir, f'{name}_frame_{frame_idx}.html')
        fig.write_html(file_path)
        print(f'Saved 3D interactive plot to {file_path}')

        # 添加文件路径到列表中
        relative_path = os.path.relpath(file_path, save_dir)
        file_list.append(relative_path.replace('\\', '/'))
    # 创建 HTML 查看器
    html_content = create_html_viewer().replace('FILE_LIST', str(file_list))
    main_html_path = os.path.join(save_dir, f'{name}_viewer.html')
    with open(main_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Saved main HTML viewer to {main_html_path}')

def get_keypoints_error(model, device, dataloader):
    model.eval()

    all_errors = []  
    all_subjects = []
    all_actions = []

    iterator = tqdm(dataloader, total=len(dataloader), desc=f"Calculating Keypoints Error")

    for batch in iterator:
        batch_root_x = batch[0]['root'].to(device)
        batch_rotations_x = batch[0]['rotations'].to(device)
        batch_bone_lengths = batch[0]['bone_lengths'].to(device)

        batch_root_y = batch[1]['root'].to(device)
        batch_rotations_y = batch[1]['rotations'].to(device)

        meta_data = batch[2]

        root_out, rotations_out = model(batch_root_x, batch_rotations_x)

        joints_predicted = rebuild_pose_from_root(root_out, rotations_out, batch_bone_lengths)
        joints_gt = rebuild_pose_from_root(batch_root_y, batch_rotations_y, batch_bone_lengths)

        errors = torch.mean(torch.abs(joints_predicted - joints_gt), dim=(1, 2, 3))  # 对 seq_len, num_joints, 3 三个维度求平均

        all_errors.append(errors.detach().cpu().numpy())
        all_subjects.extend(meta_data[:, 0].cpu().numpy())
        all_actions.extend(meta_data[:, 1].cpu().numpy())

    # 将所有批次的误差合并为一个 NumPy 数组
    all_errors = np.concatenate(all_errors)

    # 使用 Pandas DataFrame 来组织数据
    import pandas as pd
    df = pd.DataFrame({'subject': all_subjects, 'action': all_actions, 'error': all_errors})

    # 使用 groupby 计算每个 subject 和 action 的平均误差
    error_dict = df.groupby(['subject', 'action'])['error'].mean().to_dict()

    return error_dict

        


def create_html_viewer():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>3D Pose Viewer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            #header {
                position: fixed;
                top: 0;
                width: 100%;
                background-color: #f8f9fa;
                padding: 10px;
                text-align: center;
                border-bottom: 1px solid #ddd;
            }
            #content {
                margin-top: 50px;
                text-align: center;
            }
            button {
                margin: 10px;
                padding: 10px 20px;
                font-size: 16px;
            }
            iframe {
                width: 100%;
                height: 80vh;
                border: none;
            }
        </style>
    </head>
    <body>
        <div id="header">
            <span id="filename"></span>
            <button onclick="prevFrame()">上一张</button>
            <button onclick="nextFrame()">下一张</button>
        </div>
        <div id="content">
            <iframe id="viewer" src=""></iframe>
        </div>

        <script>
            const files = FILE_LIST;

            let currentIndex = 0;

            function updateViewer() {
                const viewer = document.getElementById('viewer');
                const filenameDisplay = document.getElementById('filename');
                viewer.src = files[currentIndex];
                filenameDisplay.textContent = files[currentIndex];
            }

            function prevFrame() {
                if (currentIndex > 0) {
                    currentIndex--;
                    updateViewer();
                }
            }

            function nextFrame() {
                if (currentIndex < files.length - 1) {
                    currentIndex++;
                    updateViewer();
                }
            }

            // Initialize viewer with the first file
            updateViewer();
        </script>
    </body>
    </html>
    """
    return  html_content