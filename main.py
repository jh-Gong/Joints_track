import os
import shutil
from datetime import datetime
from tqdm import tqdm

import torch
from torch import autograd
from torch import optim
from tensorboardX import SummaryWriter

from mvn.utils import cfg
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss
from mvn.models.basicnet import LstmModel
from mvn.utils.data import setup_dataloaders
import mvn.utils.misc as misc
from mvn.utils.pose_show_3d import save_3d_png

def setup_experiment(config, model_name, is_train=True):
    """
    准备实验所需的目录和日志。

    Args:
        config: 实验配置对象，包含实验的各种配置参数。
        model_name: 模型名称，用于生成实验标题。
        is_train: 布尔值，表示是否为训练模式。默认为True。

    Returns:
        experiment_dir: 实验目录路径。
        writer: TensorBoard SummaryWriter对象，用于记录训练或评估过程中的信息。
    """
    # 根据是否为训练模式，设置前缀为空或"eval_"
    prefix = "" if is_train else "eval_"

    # 根据配置中的标题和模型名称生成实验标题
    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    # 添加前缀以区分训练和评估
    experiment_title = prefix + experiment_title

    # 生成实验名称，包含标题和当前时间
    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%m-%d-%H-%M-%S.%Y"))
    print("Experiment name: {}".format(experiment_name))

    # 创建实验目录
    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 创建检查点目录，用于保存模型权重
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 将实验配置文件复制到实验目录中
    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # 创建TensorBoard SummaryWriter对象
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # 记录配置信息到TensorBoard
    writer.add_text(misc.config_to_str(config), "config", 0)

    # 返回实验目录路径和SummaryWriter对象
    return experiment_dir, writer

def one_epoch(model, criterion, opt, config, dataloader, device, epoch, scaling_info, is_train=True, experiment_dir=None, writer=None):
    """
    训练或验证模型一个epoch。

    参数:
    - model: 使用的模型。
    - criterion: 损失函数。
    - opt: 优化器。
    - config: 配置对象，包含模型和训练配置。
    - dataloader: 数据加载器。
    - device: 设备，'cuda' 或 'cpu'。
    - epoch: 当前epoch编号。
    - is_train: 布尔值，表示是否为训练模式。
    - experiment_dir: 实验目录路径，用于保存结果。
    - writer: TensorBoard writer，用于记录训练过程。

    返回:
    无
    """
    # 根据训练或验证状态设置名称
    name = "train" if is_train else "val"
    # 初始化总损失
    total_loss = 0.0

    # 根据训练或验证状态设置模型状态
    if is_train:
        model.train()
    else:
        model.eval()


    # 确认梯度计算是否使用
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        # 初始化进度条
        iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{name} Epoch {epoch + 1}/{config.opt.n_epochs}")
        # 遍历数据集中的每个批次
        for batch_idx, batch in iterator:
            with autograd.detect_anomaly():
                # 检查批次是否为空
                if batch is None:
                    print("Found None batch")
                    continue

                # 预处理batch
                batch_x = batch[0].to(device)
                batch_y = batch[1].to(device)

                # 前向传播和反向梯度计算
                outputs = model(batch_x) 
                keypoints_binary_validity = torch.ones_like(outputs)
                keypoints_binary_validity[:, -1, ...] = config.opt.pre_frame_weight
                loss = criterion(outputs, batch_y, keypoints_binary_validity)
                if is_train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # 累加损失
                total_loss += loss.item()

                # 使用TensorBoard记录损失
                if writer is not None:
                    writer.add_scalar(f'Loss/batch_{name}', loss.item(), epoch * len(dataloader) + batch_idx)
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}], Loss: {avg_loss:.10f}')

        # 记录每个epoch的平均损失
        if writer is not None:
            writer.add_scalar(f'Loss/epoch_{name}', avg_loss, epoch)

    # 定期保存3D图形
    if config.opt.save_3d_png and epoch % config.opt.save_3d_png_freq == 0 and config.opt.n_joints == 17 :
        print("Saving 3d png...")
        save_3d_png(model, device, dataloader, experiment_dir, name, epoch, scaling_info)

    
def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config
    config = cfg.load_config(args.config)

    model = {
        "lstm": LstmModel
    }[config.model.name](3 * config.opt.n_joints, config.model.n_hidden_layer, 3 * config.opt.n_joints, config.model.n_layers).to(device)

    if (config.model.init_weights):
        print("Loading pretrained weights...")
        model.load_state_dict(torch.load(config.model.checkpoint, weights_only=True))

    # criterion
    criterion_class = {
        "mse": KeypointsMSELoss,
        "mse_smooth": KeypointsMSESmoothLoss,
        "mae": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "mse_smooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    # optimizer
    opt = None
    if not args.eval:
        model_total = sum([param.nelement() for param in model.parameters()])  # 计算模型参数
        print("Number of model_total parameter: %.8fM" % (model_total / 1e6))
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_scaling_info, val_scaling_info = setup_dataloaders(config, is_train=True if not args.eval else False) 

    # experiment
    experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    if not args.eval:
        # train loop
        for epoch in range(config.opt.n_epochs):
            one_epoch(model, criterion, opt, config, train_dataloader, device, epoch, train_scaling_info, is_train=True, experiment_dir=experiment_dir, writer=writer)
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))
            one_epoch(model, criterion, opt, config, val_dataloader, device, epoch, val_scaling_info, is_train=False, experiment_dir=experiment_dir, writer=writer)

    else:
        if args.eval_dataset == 'train':
            one_epoch(model, criterion, opt, config, train_dataloader, device, 0, train_scaling_info, is_train=False, experiment_dir=experiment_dir, writer=writer)
        else:
            one_epoch(model, criterion, opt, config, val_dataloader, device, 0, val_scaling_info, is_train=False, experiment_dir=experiment_dir, writer=writer)



if __name__ == '__main__':
    work_directory = os.path.dirname(os.path.abspath(__file__))
    args = cfg.parse_args(work_directory)
    print("args: {}".format(args))
    main(args)
