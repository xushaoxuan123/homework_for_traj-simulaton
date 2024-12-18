import torch
import torch.nn as nn
from dataset import process_data, process
def get_trainer(args):
    if args.learning_strategy == 'normal_learning':
        return Normal_trainer(args)
    elif args.learning_strategy == 'curriculum_learning':
        return Curriculum_trainer(args)
    elif args.learning_strategy == 'RK4':
        return RKTrainer(args)
    elif args.learning_strategy == 'polynomial':
        return Polynomial_trainer(args)
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
            train_loss = self.train(model, train_dataset, optimizer, epoch)
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
        logger.info(f'best valid loss: {self.best}')
    def train(self, model, train_dataset, optimizer, epoch):
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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
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
        input = test_dataset['input'][:,: 0 + self.length, :self.input_size]

        for i in range(test_dataset['input'].shape[1] - self.length - self.window):
            label = test_dataset['input'][:,i + self.length: i+ self.length + self.window, :3]
            output = model(input)
            loss = criterion(output, label)
            output = process(output)
            input = output[:,-self.length:, :self.input_size]
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
    def predict(self,model, dataset, mode = 'all'):
        model.eval()
        torch.enable_grad(False)
        criterion = nn.MSELoss()
        if mode == 'subset':
            input = dataset['input'][:,:self.length, :self.input_size]
            label = dataset['input'][:,self.length:self.length + self.window, :3]
            output = model(input)
            loss = criterion(output, label)
        else:
            outputs = []
            _loss = 0
            count = 0
            input = dataset['input'][:,:self.length, :self.input_size]
            for i in range(self.window + self.length, dataset['input'].shape[1], self.window):
                output = model(input)
                outputs.append(output)
                label = dataset['input'][:,i - self.window:i, :3]
                loss = criterion(output, label)
                _loss += loss.item()
                output = process(output)
                input = output[:,-self.length:, :self.input_size]
                count += 1
            output = model(input)
            outputs.append(output[:,:30,:])
            label = dataset['input'][:,-30:, :3]
            count += 1
            loss = criterion(output[:,:30,:], label)
            label = dataset['input'][:,self.length:, :3]
            output = torch.cat(outputs, dim = 1)
            _loss += loss.item()
            loss = _loss/count
        torch.enable_grad(True)
        return output, label, loss

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

class Polynomial_trainer(Normal_trainer):
    def __init__(self,args):
        super().__init__(args)
        self.input_size = 3

class RKTrainer(Normal_trainer):
    def __init__(self, args):
        super().__init__(args)
        self.input_size = 3
    def train(self, model, train_dataset, optimizer, epoch):
        criterion = nn.MSELoss()
        _loss = 0
        count = 0
        speed = torch.zeros_like(train_dataset['input'][:,0, :self.input_size])
        for i in range(train_dataset['input'].shape[1]-self.window):
            input = train_dataset['input'][:,i, :self.input_size]
            label = train_dataset['input'][:,i + 30, :3]
            optimizer.zero_grad()
            outputs = []
            loss = 0
            for j in range(round(30)):
                time = (i+j)*0.01
                time = torch.tensor(time)
                time = time.repeat(input.shape[0],1)
                output = model(input, time)
                outputs.append(output)
                input = output.clone()
                if j == 0:
                    loss = criterion(output, train_dataset['input'][:,i+1, :3])
            loss = loss + criterion(output, label)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            _loss += loss.item()
            count += 1
            optimizer.step()
        return (_loss/(count), 0)
    def test(self, model, test_dataset):
        criterion = nn.MSELoss()
        _loss = 0
        count = 0
        speed = torch.zeros_like(test_dataset['input'][:,0, :self.input_size])
        input = test_dataset['input'][:,0, :self.input_size]
        for i in range(test_dataset['input'].shape[1]-1):
            label = test_dataset['input'][:,i + 1, :3]
            time = (i)*0.01
            time = torch.tensor(time)
            time = time.repeat(input.shape[0],1)
            output = model(input, time)
            speed = label - input
            loss = criterion(output, label)
            _loss += loss.item()
            count += 1
            input = output
        return _loss/(count)
    def predict(self,model, dataset, mode = 'all'):
        criterion = nn.MSELoss()
        if mode == 'subset':
            outputs = []
            _loss = 0
            count = 0
            input = dataset['input'][:,0, :self.input_size]
            speed = dataset['input'][:,1, :self.input_size] - dataset['input'][:,0, :self.input_size]
            for i in range(50):
                time = (i)*0.01
                time = torch.tensor(time)
                time = time.repeat(input.shape[0],1)
                output = model(input, time)
                outputs.append(output)
                speed = output - input
                label = dataset['input'][:,i + 1, :3]
                loss = criterion(output, label)
                _loss += loss.item()
                input = output
                count += 1
            label = dataset['input'][:,1:50, :3]
            output = torch.stack(outputs,dim=1)
            loss = _loss/count
        else:
            outputs = []
            _loss = 0
            count = 0
            input = dataset['input'][:,0, :self.input_size]
            speed = dataset['input'][:,1, :self.input_size] - dataset['input'][:,0, :self.input_size]
            criterion = nn.MSELoss()
            for i in range(dataset['input'].shape[1] - 1):
                time = (i)*0.01
                time = torch.tensor(time)
                time = time.repeat(input.shape[0],1)
                output = model(input, time)
                outputs.append(output)
                speed = output - input
                label = dataset['input'][:,i + 1, :3]
                loss = criterion(output, label)
                _loss += loss.item()
                input = output
                count += 1
            loss = _loss/count
            label = dataset['input'][:,1:, :3]
            output = torch.stack(outputs, dim = 1)
        return output, label, loss