import os
import pandas as pd
import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
import plotly.graph_objects as go
from typing import Dict, Union

from mvn.utils.rebuild import rebuild_pose_from_root

def save_3d_png(
    model: Module,
    device: torch.device,
    dataloader: DataLoader,
    experiment_dir: str,
    name: str,
    epoch: int
):
    """
    保存3D姿态的可视化结果为交互式HTML文件。

    Args:
        model (Module): 训练好的模型。
        device (torch.device): 'cuda' 或 'cpu'。
        dataloader (DataLoader): 数据加载器。
        experiment_dir (str): 实验目录。
        name (str): 保存文件的前缀名 (例如, 'train' 或 'val')。
        epoch (int): 当前的epoch。
    """
    model.eval()
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("数据加载器为空，无法生成可视化。")
        return

    batch_root_x = batch[0]['root'][0:1].to(device)
    batch_rotations_x = batch[0]['rotations'][0:1].to(device)
    batch_bone_lengths = batch[0]['bone_lengths'][0:1].to(device)
    batch_root_y = batch[1]['root'][0:1].to(device)
    batch_rotations_y = batch[1]['rotations'][0:1].to(device)

    with torch.no_grad():
        root_out, rotations_out = model(batch_root_x, batch_rotations_x)

    seq_len = root_out.shape[1]

    output = rebuild_pose_from_root(root_out, rotations_out, batch_bone_lengths)[0].cpu().numpy()
    batch_y = rebuild_pose_from_root(batch_root_y, batch_rotations_y, batch_bone_lengths)[0].cpu().numpy()

    connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 14), (14, 15),
        (15, 16), (8, 11), (11, 12), (12, 13)
    ]

    save_dir = os.path.join(experiment_dir, 'pose_3d', f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    file_list = []
    for frame_idx in range(seq_len):
        scatter_gt = go.Scatter3d(
            x=batch_y[frame_idx, :, 0], y=batch_y[frame_idx, :, 1], z=batch_y[frame_idx, :, 2],
            mode='markers', marker=dict(size=5, color='green'), name='Ground Truth'
        )
        scatter_pred = go.Scatter3d(
            x=output[frame_idx, :, 0], y=output[frame_idx, :, 1], z=output[frame_idx, :, 2],
            mode='markers', marker=dict(size=5, color='red'), name='Prediction'
        )

        lines = []
        for start, end in connections:
            lines.append(go.Scatter3d(
                x=[batch_y[frame_idx, start, 0], batch_y[frame_idx, end, 0]],
                y=[batch_y[frame_idx, start, 1], batch_y[frame_idx, end, 1]],
                z=[batch_y[frame_idx, start, 2], batch_y[frame_idx, end, 2]],
                mode='lines', line=dict(color='green', width=2), showlegend=False
            ))
            lines.append(go.Scatter3d(
                x=[output[frame_idx, start, 0], output[frame_idx, end, 0]],
                y=[output[frame_idx, start, 1], output[frame_idx, end, 1]],
                z=[output[frame_idx, start, 2], output[frame_idx, end, 2]],
                mode='lines', line=dict(color='red', width=2), showlegend=False
            ))

        all_points = np.concatenate((batch_y[frame_idx], output[frame_idx]), axis=0)
        min_vals, max_vals = all_points.min(axis=0), all_points.max(axis=0)
        center = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() * 0.6

        layout = go.Layout(
            title=f'Frame {frame_idx}',
            scene=dict(
                xaxis=dict(range=[center[0] - max_range, center[0] + max_range]),
                yaxis=dict(range=[center[1] - max_range, center[1] + max_range]),
                zaxis=dict(range=[center[2] - max_range, center[2] + max_range]),
                aspectmode='cube'
            )
        )

        fig = go.Figure(data=[scatter_gt, scatter_pred] + lines, layout=layout)
        file_path = os.path.join(save_dir, f'{name}_frame_{frame_idx}.html')
        fig.write_html(file_path)
        file_list.append(os.path.relpath(file_path, save_dir).replace('\\', '/'))

    html_content = create_html_viewer(file_list)
    main_html_path = os.path.join(save_dir, f'{name}_viewer.html')
    with open(main_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"主HTML查看器已保存至: {main_html_path}")

def get_keypoints_error(model: Module, device: torch.device, dataloader: DataLoader) -> Dict:
    """
    计算并汇总关键点在不同subjects和actions上的平均误差。

    Args:
        model (Module): 训练好的模型。
        device (torch.device): 'cuda' 或 'cpu'。
        dataloader (DataLoader): 数据加载器。

    Returns:
        Dict: 包含按 subject 和 action 分类的关键点误差的字典。
    """
    model.eval()
    all_errors, all_subjects, all_actions = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="正在计算关键点误差"):
            batch_root_x = batch[0]['root'].to(device)
            batch_rotations_x = batch[0]['rotations'].to(device)
            batch_bone_lengths = batch[0]['bone_lengths'].to(device)
            batch_root_y = batch[1]['root'].to(device)
            batch_rotations_y = batch[1]['rotations'].to(device)

            root_out, rotations_out = model(batch_root_x, batch_rotations_x)
            joints_predicted = rebuild_pose_from_root(root_out, rotations_out, batch_bone_lengths)
            joints_gt = rebuild_pose_from_root(batch_root_y, batch_rotations_y, batch_bone_lengths)

            errors = torch.norm(joints_predicted - joints_gt, p=2, dim=-1).mean(dim=(-1, -2))
            all_errors.append(errors.cpu().numpy())
            all_subjects.extend(batch[2][0])
            all_actions.extend(batch[2][1])

    df = pd.DataFrame({
        'subject': all_subjects,
        'action': all_actions,
        'error': np.concatenate(all_errors)
    })

    error_df = df.groupby(['subject', 'action'])['error'].agg(['mean', 'count']).reset_index()

    subject_avg = {
        subject: np.average(group['mean'], weights=group['count'])
        for subject, group in error_df.groupby('subject')
    }

    error_dict = error_df.groupby(['subject', 'action'])['mean'].mean().unstack(0).to_dict()
    for subject, avg in subject_avg.items():
        if subject in error_dict:
            error_dict[subject]['average'] = avg

    all_actions_avg = {
        action: np.average(group['mean'], weights=group['count'])
        for action, group in error_df.groupby('action')
    }
    all_actions_avg['overall_average'] = np.average(error_df['mean'], weights=error_df['count'])
    error_dict['average'] = all_actions_avg

    return error_dict

def create_html_viewer(file_list: list) -> str:
    """
    为一系列HTML文件创建一个主查看器页面。

    Args:
        file_list (list): 要在查看器中引用的HTML文件名列表。

    Returns:
        str: 包含完整HTML内容的字符串。
    """
    files_json = json.dumps(file_list)
    return f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>3D姿态查看器</title>
        <style>
            body {{ font-family: sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; }}
            #header {{ flex: 0 0 auto; padding: 10px; text-align: center; background-color: #f0f0f0; border-bottom: 1px solid #ccc; }}
            #content {{ flex: 1 1 auto; }}
            iframe {{ width: 100%; height: 100%; border: none; }}
            button {{ margin: 5px; padding: 10px; font-size: 16px; }}
        </style>
    </head>
    <body>
        <div id="header">
            <span id="filename"></span>
            <button onclick="navigate(-1)">上一帧</button>
            <button onclick="navigate(1)">下一帧</button>
        </div>
        <div id="content">
            <iframe id="viewer"></iframe>
        </div>
        <script>
            const files = {files_json};
            let currentIndex = 0;
            const viewer = document.getElementById('viewer');
            const filenameDisplay = document.getElementById('filename');

            function updateViewer() {{
                if (files.length > 0) {{
                    const file = files[currentIndex];
                    viewer.src = file;
                    filenameDisplay.textContent = `文件: ${{file}} (${{currentIndex + 1}}/${{files.length}})`;
                }}
            }}
            function navigate(direction) {{
                const newIndex = currentIndex + direction;
                if (newIndex >= 0 && newIndex < files.length) {{
                    currentIndex = newIndex;
                    updateViewer();
                }}
            }}
            updateViewer();
        </script>
    </body>
    </html>
    """
