import torch
import torch.nn as nn
from dataset import process_data, process
def get_trainer(args):
    if args.learning_strategy == 'normal_learning':
        return Normal_trainer(args)
    elif args.learning_strategy == 'curriculum_learning':
        return Curriculum_trainer(args)
    else:
        raise ValueError('Invalid learning strategy')
    
class Normal_trainer:
    def __init__(self,args):
        self.alpha = args.alpha
        self.epoch = args.epochs
        self.strategy = args.target_strategy
        self.length = args.length
        self.window = args.window
        self.best = torch.tensor(float('inf'))
        self.model_name = args.model
        self.input_size = args.input_size   
        self.train_loss = []
        self.valid_loss = []
        self.save_path = args.save_path
    def fit(self,model, train_dataset, test_dataset, optimizer, scheduler, logger):
        for epoch in range(self.epoch):
            train_loss = self.train(model, train_dataset, optimizer)
            print('epoch:', epoch)
            print('train_loss:', train_loss)
            logger.info(f'epoch: {epoch}, train_loss: {train_loss}')
            self.train_loss.append(train_loss)
            valid_loss = self.test(model, test_dataset)
            print('valid_loss:', valid_loss)
            logger.info(f'epoch: {epoch}, valid_loss: {valid_loss}')
            self.valid_loss.append(valid_loss)
            if valid_loss < self.best:
                self.best = valid_loss
                self.save(model)
            scheduler.step()
        self.save_loss()
    def train(self, model, train_dataset, optimizer):
        model.train()
        criterion = nn.MSELoss()  # 先创建MSELoss实例
        _loss = 0
        _loss_a = 0
        count = 0
        for i in range(train_dataset['input'].shape[1] - self.length - self.window):
            input = train_dataset['input'][:,i: i + self.length, :self.input_size]
            label = train_dataset['input'][:,i + self.length: i+ self.length + self.window, :3]
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            _loss += loss.item()    
            if self.strategy == 'multi_target_learing':
                loss_a = self.alpha * torch.sum((1- torch.cosine_similarity(output - train_dataset['input'][:,i + self.length - 1: i+ self.length + self.window - 1, :3], train_dataset['input'][:,i+ self.length : i+ self.length+self.window,-3:], dim = 1)))
                loss += loss_a
                _loss_a += loss_a.item()
            loss.backward()
            optimizer.step()
            count += 1
        return _loss/(count), _loss_a/(count)
    def test(self, model, test_dataset):
        model.eval()
        torch.enable_grad(False)
        criterion = nn.MSELoss()  
        input = test_dataset['input'][:,:self.length, :]
        _loss = 0
        count = 0
        for i in range(test_dataset['input'].shape[1] - self.length - self.window):
            input = test_dataset['input'][:,i: i + self.length, :self.input_size]
            label = test_dataset['input'][:,i + self.length: i+ self.length + self.window, :3]
            output = model(input)
            loss = criterion(output, label)
            output = process(output)
            # input = torch.cat((input[:,self.window:,:], output), dim = 1)
            _loss += loss.item()    
            count += 1
        torch.enable_grad(True)
        return _loss/(count)
    def save(self,model):
        state = model.state_dict()
        torch.save(state,  self.save_path + f'/{self.model_name}_model.pth')
    def save_loss(self):
        torch.save(self.train_loss, self.save_path + '/train_loss.pth')
        torch.save(self.valid_loss, self.save_path + '/valid_loss.pth')
    def load(self,model):
        model.load_state_dict(torch.load(self.save_path + f'/{self.model_name}_model.pth'))
        self.train_loss = torch.load(self.save_path + '/train_loss.pth') 
        self.valid_loss = torch.load(self.save_path + '/valid_loss.pth')
    def predict(self,model, dataset):
        model.eval()
        torch.enable_grad(False)
        input = dataset['input'][:,:self.length, :self.input_size]
        label = dataset['input'][:,self.length:self.length + self.window, :3]
        output = model(input)
        torch.enable_grad(True)
        return output, label

class Curriculum_trainer():
    def __init__(self, model, train_dataset, test_dataset, target_strategy, learing_strategy):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.target_strategy = target_strategy
        self.learing_strategy = learing_strategy
    def train(self):
        pass
    def test(self):
        pass
    def save(self):
        pass
    def load(self):
        pass