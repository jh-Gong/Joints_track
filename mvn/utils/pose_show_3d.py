import os
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go

def save_3d_png(model, device, dataloader, experiment_dir, name, epoch, scaling_info):
    batch = next(iter(dataloader))
    batch_x, batch_y, _ = batch
    batch_x = batch_x[0]    # size: (seq_len, num_joints * 3)
    batch_y = batch_y[0]    # size: (seq_len, num_joints * 3)
    batch_x = batch_x.unsqueeze(0)
    output = model(batch_x.to(device))  # size: (1, seq_len, num_joints * 3)
    output = output.squeeze(0)

    seq_len, num_joints_times_3 = batch_y.shape
    num_joints = num_joints_times_3 // 3

    batch_y = batch_y.view(seq_len, num_joints, 3).detach().cpu().numpy()
    output = output.view(seq_len, num_joints, 3).detach().cpu().numpy()

    # 反归一化或反缩放
    if scaling_info["mode"] == "norm":
        mean = scaling_info["mean"].reshape(1, num_joints, 3)
        std = scaling_info["std"].reshape(1, num_joints, 3)
        batch_y = batch_y * std + mean
        output = output * std + mean

    elif scaling_info["mode"] == "linear":
        min_val = scaling_info["min"].reshape(1, num_joints, 3)
        max_val = scaling_info["max"].reshape(1, num_joints, 3)
        batch_y = (batch_y + 1) * (max_val - min_val) / 2 + min_val
        output = (output + 1) * (max_val - min_val) / 2 + min_val

    # 关节连接方式
    connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 14), (14, 15),
        (15, 16), (8, 11), (11, 12), (12, 13)
    ]

    save_dir = os.path.join(experiment_dir, 'pose_3d', f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    # for frame_idx in range(seq_len):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.view_init(elev=90, azim=90)

    #     # ground truth 绿色
    #     for start, end in connections:
    #         ax.plot(
    #             [batch_y[frame_idx, start, 0], batch_y[frame_idx, end, 0]],
    #             [batch_y[frame_idx, start, 1], batch_y[frame_idx, end, 1]],
    #             [batch_y[frame_idx, start, 2], batch_y[frame_idx, end, 2]],
    #             'g-'
    #         )

    #     # prediction 红色
    #     for start, end in connections:
    #         ax.plot(
    #             [output[frame_idx, start, 0], output[frame_idx, end, 0]],
    #             [output[frame_idx, start, 1], output[frame_idx, end, 1]],
    #             [output[frame_idx, start, 2], output[frame_idx, end, 2]],
    #             'r-'
    #         )

    #     all_points = np.concatenate((batch_y[frame_idx], output[frame_idx]), axis=0)
    #     x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    #     y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    #     z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    #     # 空余比例
    #     margin_ratio = 0.2
    #     x_margin = (x_max - x_min) * margin_ratio
    #     y_margin = (y_max - y_min) * margin_ratio
    #     z_margin = (z_max - z_min) * margin_ratio
    #     ax.set_xlim([x_min - x_margin, x_max + x_margin])
    #     ax.set_ylim([y_min - y_margin, y_max + y_margin])
    #     ax.set_zlim([z_min - z_margin, z_max + z_margin])

    #     ax.set_title(f'Frame {frame_idx}')
        
    #     # Save the plot
    #     file_path = os.path.join(save_dir, f'{name}_frame_{frame_idx}.png')
    #     plt.savefig(file_path)
    #     plt.close(fig)

    #     print(f'Saved 3D plot to {file_path}')

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
    main_html_path = os.path.join(save_dir, 'viewer.html')
    with open(main_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Saved main HTML viewer to {main_html_path}')

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