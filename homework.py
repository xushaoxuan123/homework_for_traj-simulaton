import torch 
import torch.nn as nn
from dataset import get_dataset, process_data
train_dataset, test_dataset = get_dataset()
train_dataset = process_data(train_dataset)
test_dataset = process_data(test_dataset)
target_strategy = 'multi_target_learing'
learing_strategy = 'curriculum_learning'
model = 
