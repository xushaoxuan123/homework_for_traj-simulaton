import torch
import random
import numpy as np
import torch.nn as nn
def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(args,model):
    if args.optimizer == 'adam':
        return torch.optim.Adam(params = model.parameters() ,lr = args.learning_rate)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(params=model.parameters(),lr = args.learning_rate)
    else:
        raise ValueError('Invalid optimizer')

def get_scheduler(args, optimizer):
    if args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=True)
    else:
        raise ValueError('Invalid scheduler')
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_loss(train_loss, valid_loss, save_path=None):
    """
    train_loss: list - 训练集损失
    valid_loss: list - 验证集损失
    """
    plt.figure(figsize=(10, 5))
    train_loss, cosine_loss = [i[0] for i in train_loss], [i[1] for i in train_loss]
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(cosine_loss, label='Train Cosine_Similarity Loss', color='green')
    plt.plot(valid_loss, label='Valid Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + '/loss.png')
    plt.show()
    plt.close()

def plot_trajectories(pred, true, save_path=None, if_show=True, mode='all'):
   """
   pred: (5, 20, 3) - 预测值
   true: (5, 20, 3) - 真实值
   """
   # 创建一个图形对象，包含5个子图
   fig = plt.figure(figsize=(20, 10))
   
   for i in range(pred.shape[0]):  # 遍历每条轨迹
       # 创建3D子图
       ax = fig.add_subplot(2, 3, i+1, projection='3d')
       
       # 提取当前轨迹的预测值和真实值
       pred_traj = pred[i]  # (20, 3)
       true_traj = true[i]  # (20, 3)
       
       # 绘制预测轨迹
       ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
               'r-', label='Predicted')
               
       # 绘制真实轨迹
       ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 
               'b--', label='Ground Truth')
       
       # 添加标题和图例
       ax.set_title(f'Trajectory {i+1} {mode}')
       ax.legend()
       
       # 添加坐标轴标签
       ax.set_xlabel('X')
       ax.set_ylabel('Y')
       ax.set_zlabel('Z')
   
   plt.tight_layout()
   plt.savefig(save_path + f'/result_{mode}.png')
   plt.show()
   plt.close()

def plot_trajectories_2d(pred_traj, true_traj, save_path=None, if_show=True, mode='all'):
    """
    在一张图上绘制单个轨迹的所有坐标预测值和真值
    
    参数:
    pred_traj: numpy array - 预测轨迹, shape (sequence_length, 3)
    true_traj: numpy array - 真实轨迹, shape (sequence_length, 3)
    save_path: str - 保存路径
    traj_idx: int - 轨迹索引，用于文件命名
    """
    trajs = len(pred_traj)
    for traj_idx in range(trajs):
        plt.figure(figsize=(10, 6))
        
        coords = ['X', 'Y', 'Z']
        line_styles = ['--', '-'] 
        colors = ['blue', 'red', 'green']  # 每个坐标轴用不同颜色
        
        # 绘制预测值和真值
        for i in range(3):  # 三个坐标
            # 绘制预测值
            plt.plot(pred_traj[traj_idx,:, i], line_styles[0], 
                    color=colors[i], label=f'Predicted {coords[i]}',
                    linewidth=2)
            # 绘制真值
            plt.plot(true_traj[traj_idx,:, i], line_styles[1], 
                    color=colors[i], label=f'Ground Truth {coords[i]}',
                    linewidth=2)
        
        plt.title(f'Trajectory {traj_idx + 1}_{mode}')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.grid(True)
        plt.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path is not None:
            save_file = save_path+ f'/trajectory_{traj_idx + 1}_{mode}.png'
            plt.savefig(save_file, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
