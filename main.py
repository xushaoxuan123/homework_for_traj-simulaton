import torch 
import torch.nn as nn
from dataset import get_dataset, process_data
from model import get_model
from trainer import get_trainer
from utils import random_seed, get_optimizer, get_scheduler, plot_trajectories, plot_loss
import argparse
import os
import logging
def parse_args():
    parser = argparse.ArgumentParser(description='Your program description')
    
    # 添加参数
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train')
    parser.add_argument('--learning_strategy', type=str, default='normal_learning',
                        help='learning strategy for training')
    parser.add_argument('--target_strategy', type=str, default='multi_target_learing',
                        help='target strategy for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training')
    parser.add_argument('--model', type=str, default='lstm',
                        help='model for training')
    parser.add_argument('--input_size', type=int, default=87,
                        help='input size for model')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='sgd',)
    parser.add_argument('--scheduler', type=str, default='step',)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--length', type=int, default=20)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--if_ploynomia', type=bool, default=True)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    save_path = './results/'+ args.model + '_' + str(args.learning_rate) + '_' +  str(args.alpha) +'_' + str(args.length) + '_' + str(args.window)  + '_' + args.learning_strategy + '_' + args.target_strategy 
    args.save_path = save_path  
    random_seed(args.seed)
    model = get_model(args)
    train_dataset, test_dataset = get_dataset()
    train_dataset = process_data(train_dataset)
    test_dataset = process_data(test_dataset)
    optimizer = get_optimizer(args, model )
    scheduler = get_scheduler(args, optimizer)
    trainer = get_trainer(args)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(save_path + '/log.txt')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f'Start training with save path {save_path}')
    logger.info(f'Arguments: {args}')
    if args.mode == 'train':
        trainer.fit(model, train_dataset, test_dataset, optimizer, scheduler, logger)
    trainer.load(model)
    prediction, label = trainer.predict(model, test_dataset)
    prediction = prediction.detach().numpy()
    label = label.detach().numpy()
    plot_loss(trainer.train_loss, trainer.valid_loss,save_path)
    plot_trajectories(prediction, label, save_path)

