import torch
import torch.nn as nn
import scipy.io as sio

def get_dataset():
    # 读取.mat文件
    mat_data = sio.loadmat('data.mat')  # 返回字典

    train_data = mat_data['trajectory_train']  # 获取训练数据
    test_data = mat_data['trajectory_test']  # 获取测试数据

    train_data = torch.from_numpy(train_data).float()
    test_data = torch.from_numpy(test_data).float()
    train_dataset = {}
    test_dataset = {}
    train_data = train_data.permute(2,1,0)
    test_data = test_data.permute(2,1,0)
    train_dataset['input'] = train_data[:, :-1, :]
    train_dataset['label'] = train_data[:, -1, :]
    test_dataset['label'] = test_data[:, -1, :]
    test_dataset['input'] = test_data[:, :-1, :]    
    # train_dataset['label'] = train_data[:, 1:, :]
    return train_data, test_data

def process_data(dataset):
    # 原本xyz三维太小，人为增加特征数量
    for i in range(dataset['input'].shape[0]):
        if i > 0:
            speed = dataset['input'][i] - dataset['input'][i-1]
        else:
            speed = torch.zeros_like(dataset['input'][i])
        if i < dataset['input'].shape[0]-1:
            speed_next = dataset['input'][i+1] - dataset['input'][i]
        else:
            speed_next = torch.zeros_like(dataset['input'][i])
        dataset['input'][i] = torch.cat((dataset['input'][i], speed), dim=1) #speed
        dataset['speed_label'][i] = speed_next ## speed label to help train
    return dataset