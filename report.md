# 人工智能实践第三次作业

学号：`2022200648`

姓名： ` 徐少轩`

## 作业描述

使用20预测50的setting进行轨迹预测。测试时使用测试集部分，分别对20预测50和20预测980两个任务进行测试和可视化处理。对于RK4的方法，由于方法限制是依次预测（1预测1），所以使用的是1预测10和1预测999的setting。

## 模型实现

代码中实现了lstm，transformer_encoder，多项式回归，四阶龙格库塔模型。代码随附在了作业中，同时可以在[github仓库](https://github.com/xushaoxuan123/homework_for_traj-simulaton)中找到。

### Input

所有模型的输入shape均为`(traj_nums(batchsize), seq_len, input_size)`，输出均为`(traj_nums(batchsize), output_len, 3)` 。对于不同模型，输入的inputsize存在不同。lstm和transformer_encoder使用的输入如下：

```python
def process_data(dataset):
    # 原本xyz三维太小，人为增加特征数量
    new_dataset = defaultdict(list)
    for i in range(dataset['input'].shape[0]):
        polynomia_feature = torch.cat((dataset['input'][i], dataset['input'][i]**2, dataset['input'][i]**3), dim=1)
        polynomia_feature = polynomia_feature.unsqueeze(2)
        polynomia_feature = torch.bmm(polynomia_feature, polynomia_feature.permute(0,2,1)).squeeze(2)
        polynomia_feature = polynomia_feature.flatten(1,2)
        polynomia_feature = torch.cat([dataset['input'][i], dataset['input'][i]**2, dataset['input'][i]**3, polynomia_feature], dim=1)
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
```

增加了xyz的高次项以及两两交叉项。

对于多项式回归，出于对训练环节的考虑，使用

```python
polynomia_feature = torch.cat((x, x**2, x**3), dim=1)
```

对于RK4而言，添加了估计的速度项作为输入

```python
def process_data(self, x,k):
        polynomia_feature = process(x)
        polynomia_feature = torch.cat((polynomia_feature, k), dim=1)
        return polynomia_feature
```

### ouput

对于RK4方法，采用1预测1的格式。对于剩下几种方法，均是将长度为seq_len的输入映射到output_len。

### 模型结构

#### transformer_encoder

```python
class Transformer_model_linear(nn.Module):
    def __init__(self, args):
        super(Transformer_model_linear, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.input_size,
                nhead=3,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.linear = nn.Linear(args.input_size, 3)
        self.fc = nn.Linear(args.length, args.window)  
        self.output_len = args.window
    def forward(self, x):
        output = self.transformer(x)
        output = output.permute(0, 2, 1)
        output = self.fc(output)
        output = output.permute(0, 2, 1)
        output = self.linear(output)
        return output
```

#### lstm

```python
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(
            embed_dim=args.hidden_size,
            num_heads=4,
            batch_first=True
        )
        self.query_rnn = nn.GRU(
            args.hidden_size, 
            args.hidden_size,
            batch_first=True
        )
        self.linear = nn.Linear(args.hidden_size, 3)
        self.output_len = args.window   
        self.hidden_size = args.hidden_size
        self.linear_out = nn.Linear(args.length, args.window)
    def forward(self,x):
        batch_size = x.size(0)
        output, (h_n,c_n) = self.lstm(x)
        output = output.permute(0, 2, 1)
        output = self.linear_out(output)
        output = output.permute(0, 2, 1)
        output = self.linear(output)
        return output
```

#### RK4

```python
#4阶龙格库塔
class RungeKutta_4(nn.Module):
    def __init__(self, args):
        super(RungeKutta_4, self).__init__()
        self.linear = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 3))
        self.h = args.h
    def process_data(self, x,k):
        polynomia_feature = torch.cat([x, x**2, x**3],dim = 1)
        polynomia_feature = torch.cat((polynomia_feature, k), dim=1) ##考虑当前速度作为下一时刻位置的变量
        return polynomia_feature
    def forward(self, x, speed):
        k1 = self.linear(self.process_data(x, speed))
        # k2 = self.linear(self.process_data(x[:,:3] + self.h/2 * k1, k1*2))
        # k3 = self.linear(self.process_data(x[:,:3] + self.h/2 * k2, k2*2))
        # k4 = self.linear(self.process_data(x[:,:3] + self.h * k3, k3))
        k2 = self.linear(self.process_data(x + self.h/2 * k1, 2 * k1))
        k3 = self.linear(self.process_data(x + self.h/2 * k2, 2 * k2))
        k4 = self.linear(self.process_data(x + self.h * k3, k3))
        return x + self.h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
def get_model(args):
    if args.model == 'lstm':
        return Model(args)
    elif args.model == 'transformer':
        # transformer
        return Transformer_model(args)
    elif args.model == 'transformer_linear':
        return Transformer_model_linear(args)
    elif args.model == 'rk4':
        return RungeKutta_4(args)
    elif args.model == 'linear':
        return Polynomial_model(args)
```

#### 多项式回归

```python
class Polynomial_model(nn.Module):
    def __init__(self, args):
        super(Polynomial_model, self).__init__()
        self.linear = nn.Sequential(nn.Linear(9 * args.length, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 3 * args.window))
        self.window = args.window
    def forward(self, x):
        polynomia_feature = torch.cat((x, x**2, x**3), dim=1)
        polynomia_feature = polynomia_feature.flatten(1,2)
        output = self.linear(polynomia_feature)
        output = output.view(-1, self.window, 3)
        return output
```

## 训练流程

对于除了RK4的模型，使用同样的训练流程Normal_Trainer

```python
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
        logger.info(f'best valid loss: {self.best}')
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
```

对于RK4的模型，主要就是将20预测50的训练/测试架构改为了1预测1的训练/测试架构，具体代码过长就不在报告中展示了。

## 实验结果分析

由于实验setting不同，此处主要做对模型预测结果的可视化分析（例如RK4的test loss在1预测1的情况下极低，但是实际效果很差）

### training setting

使用SGD优化器，多项式回归和RK4使用`lr =0.007`，剩下的使用`lr = 0.1`。其余setting与代码中默认相同

### transformer_encoder

![result_subset](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/transformer_linear_0.1_0.0_sgd_20_50_normal_learning_multi_target_learing/result_subset.png)

![result_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/transformer_linear_0.1_0.0_sgd_20_50_normal_learning_multi_target_learing/result_all.png)

第一张图是测试集上三条轨迹20-70的数据，可以看到transformer能够较好的在20预测50的情况下拟合函数走向，在第二个轨迹上拟合的效果更好。第二张图代表的是20预测980的结果，预测结果由于误差的不断累积与真实轨迹几乎不贴合。



### lstm

![result_subset](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/lstm_0.1_0.0_sgd_20_50_normal_learning_multi_target_learing/result_subset.png)

![result_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/lstm_0.1_0.0_sgd_20_50_normal_learning_multi_target_learing/result_all.png)

可以看到lstm在20预测50的情况下，在第二条轨迹上拟合的效果强于trasnformer，但是1，3条的效果较差，同时可以发现轨迹的光滑程度不如transformer。

此外，观察20预测980的任务，可以发现预测的轨迹有一定的贴合效果，能看出和真实轨迹趋势上的相似。

### RK4

RK4使用的1预测1的预测模式，因此效果较差

![result_subset](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/rk4_0.007_0.0_sgd_20_50_RK4_multi_target_learing/result_subset.png)

![result_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/rk4_0.007_0.0_sgd_20_50_RK4_multi_target_learing/result_all.png)

可以看到两种任务上表现都不佳，但是在整体轨迹预测上则呈现出了一条光滑的轨迹，可能的原因是RK4方法在拟合时需要提供微分方程的边界条件，而数据中难以给出，因此学到了一个简单的周期轨迹。

### 多项式回归

![result_subset](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/linear_0.007_0.0_sgd_20_50_polynomial_multi_target_learing/result_subset.png)

![result_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/linear_0.007_0.0_sgd_20_50_polynomial_multi_target_learing/result_all.png)

多项式回归预测效果很差，可以发现预测的轨迹波动程度很大，平滑性也不行。虽然网络和RK4使用的是几乎相同setting的，但是RK4多次迭代的效果使得预测轨迹更加平滑。此外，由于输入上对lstm和transformer做了同样，乃至更多的特征处理，简单的多项式回归比不过更复杂的网络是十分合理的。

由于图数量过多，单个维度的可视化在result文件夹中上交，报告中暂且忽略

## TODO & Hypothesis

1. RK4使用更多的数据处理方式，尝试解决边界条件带来的问题（以及验证是否是该问题导致的）
2. 先前和助教师兄讨论过的sin trick还未尝试
3. 尝试更多的setting，50测50仅进行过少量测试，效果如下

### lstm

![result_subset](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/lstm_0.1_0.0_sgd_50_50_normal_learning_multi_target_learing/result_subset.png)

![trajectory_1_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/lstm_0.1_0.0_sgd_50_50_normal_learning_multi_target_learing/trajectory_1_all.png)

![trajectory_2_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/lstm_0.1_0.0_sgd_50_50_normal_learning_multi_target_learing/trajectory_2_all.png)

![trajectory_3_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/lstm_0.1_0.0_sgd_50_50_normal_learning_multi_target_learing/trajectory_3_all.png)

![result_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/lstm_0.1_0.0_sgd_50_50_normal_learning_multi_target_learing/result_all.png)



### RK4

![result_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/rk4_0.007_0.0_sgd_50_50_RK4_multi_target_learing/result_all.png)

![result_subset](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/rk4_0.007_0.0_sgd_50_50_RK4_multi_target_learing/result_subset.png)

![trajectory_1_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/rk4_0.007_0.0_sgd_50_50_RK4_multi_target_learing/trajectory_1_all.png)

![trajectory_2_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/rk4_0.007_0.0_sgd_50_50_RK4_multi_target_learing/trajectory_2_all.png)

![trajectory_3_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/rk4_0.007_0.0_sgd_50_50_RK4_multi_target_learing/trajectory_3_all.png)

### 多项式回归

![result_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/linear_0.007_0.0_sgd_50_50_polynomial_multi_target_learing/result_all.png)

![result_subset](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/linear_0.007_0.0_sgd_50_50_polynomial_multi_target_learing/result_subset.png)

![trajectory_1_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/linear_0.007_0.0_sgd_50_50_polynomial_multi_target_learing/trajectory_1_all.png)

![trajectory_2_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/linear_0.007_0.0_sgd_50_50_polynomial_multi_target_learing/trajectory_2_all.png)

![trajectory_3_all](/Users/writemath/课程学习/大学课件/大三上/人工智能实践/作业3/results/linear_0.007_0.0_sgd_50_50_polynomial_multi_target_learing/trajectory_3_all.png)