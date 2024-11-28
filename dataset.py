import torch
import torch.nn as nn
import scipy.io as sio
from collections import defaultdict

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
    train_dataset['input'] = train_data
    test_dataset['input'] = test_data 
    # train_dataset['label'] = train_data[:, 1:, :]
    return train_dataset, test_dataset

def process_data(dataset):
    # 原本xyz三维太小，人为增加特征数量
    new_dataset = defaultdict(list)
    for i in range(dataset['input'].shape[0]):
        polynomia_feature = torch.cat((dataset['input'][i], dataset['input'][i]**2, dataset['input'][i]**3), dim=1)
        polynomia_feature = polynomia_feature.unsqueeze(2)
        polynomia_feature = torch.bmm(polynomia_feature, polynomia_feature.permute(0,2,1)).squeeze(2)
        polynomia_feature = polynomia_feature.flatten(1,2)
        polynomia_feature = torch.cat([dataset['input'][i], polynomia_feature], dim=1)
        if i > 0:
            speed = dataset['input'][i] - dataset['input'][i-1]
        else:
            speed = torch.zeros_like(dataset['input'][i])
        if i < dataset['input'].shape[0]-1:
            speed_next = dataset['input'][i+1] - dataset['input'][i]
        else:
            speed_next = torch.zeros_like(dataset['input'][i])
        new_dataset['input'].append(torch.cat((polynomia_feature ,speed), dim=-1)) #speed
        new_dataset['speed_label'].append(speed_next) ## speed label to help train
    new_dataset['input'] = torch.stack(new_dataset['input'])
    new_dataset['speed_label'] = torch.stack(new_dataset['speed_label'])
    return new_dataset

def process(dataset):
    new_dataset = []
    for i in range(dataset.shape[0]):
        polynomia_feature = torch.cat((dataset[i], dataset[i]**2, dataset[i]**3), dim=1)
        polynomia_feature = polynomia_feature.unsqueeze(2)
        polynomia_feature = torch.bmm(polynomia_feature, polynomia_feature.permute(0,2,1)).squeeze(2)
        polynomia_feature = polynomia_feature.flatten(1,2)
        polynomia_feature = torch.cat([dataset[i], polynomia_feature], dim=1)
        if i > 0:
            speed = dataset[i] - dataset[i-1]
        else:
            speed = torch.zeros_like(dataset[i])
        if i < dataset.shape[0]-1:
            speed_next = dataset[i+1] - dataset[i]
        else:
            speed_next = torch.zeros_like(dataset[i])
        new_dataset.append(torch.cat((polynomia_feature ,speed), dim=-1)) #speed
    new_dataset = torch.stack(new_dataset)
    return new_dataset