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
def plot_trajectories(pred, true, save_path=None):
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
       ax.set_title(f'Trajectory {i+1}')
       ax.legend()
       
       # 添加坐标轴标签
       ax.set_xlabel('X')
       ax.set_ylabel('Y')
       ax.set_zlabel('Z')
   
   plt.tight_layout()
   plt.savefig(save_path + '/result.png')
   plt.show()
   plt.close()
