import torch
import torch.nn as nn
import math
from dataset import process
def generate_square_subsequent_mask( sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class LSTM_Model(nn.Module):
    def __init__(self, args):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=args.num_layers, batch_first=True)
        self.linear = nn.Linear(args.hidden_size, 3)
        self.output_len = args.window   
        self.hidden_size = args.hidden_size
        self.linear_out = nn.Linear(args.length, args.window)
        self.attn = nn.MultiheadAttention(args.hidden_size, 1)
    def forward(self,x):
        output, (h_n,c_n) = self.lstm(x)
        outputs = [output[:,-1,:]]
        outputs = torch.stack(outputs, dim=1)
        for i in range(self.output_len):
            output, (h_n,c_n) = self.decoder(outputs)
            outputs = torch.cat((outputs, output[:,-1,:].unsqueeze(1)), dim=1)
        output = self.linear(outputs)
        return output[:,1:,:]

class Transformer_model(nn.Module):
    def __init__(self, args):
        super(Transformer_model, self).__init__()
        self.transformer = nn.Transformer(
            d_model=args.input_size,
            nhead=1,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.input_size = args.input_size
        self.linear = nn.Linear(args.input_size, 3)
        self.output_len = args.window
    def forward(self, x, label):
        batch_size = x.size(0)        
        if label == None:
            tgt = torch.zeros(batch_size, self.output_len, self.input_size)
            x = self.positional_encoding(x)

            output = self.transformer(x, x)
            output = output.permute(0, 2, 1)
            output = self.linear(output)
        else:
            x = self.positional_encoding(x)
            output = self.transformer(x, label[:, :-1, :])
            output = self.linear(output)
        return output
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
#4阶龙格库塔
class RungeKutta_4(nn.Module):
    def __init__(self, args):
        super(RungeKutta_4, self).__init__()
        self.linear = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        self.h = args.h
    def process_data(self, x, t):
        polynomia_feature = torch.cat((x, t), dim=1)
        return x
    def forward(self, x, t):
        k1 = self.linear(self.process_data(x,t))
        # k2 = self.linear(self.process_data(x[:,:3] + self.h/2 * k1, k1*2))
        # k3 = self.linear(self.process_data(x[:,:3] + self.h/2 * k2, k2*2))
        # k4 = self.linear(self.process_data(x[:,:3] + self.h * k3, k3))
        k2 = self.linear(self.process_data(x + self.h/2 * k1, t + self.h/2))
        k3 = self.linear(self.process_data(x + self.h/2 * k2, t + self.h/2))
        k4 = self.linear(self.process_data(x + self.h * k3, t + self.h))
        return x + self.h/6 * (k1 + 2 * k2 + 2 * k3 + k4)

class Trival_model(RungeKutta_4):
    def __init__(self, args):
        super(Trival_model, self).__init__(args)
        self.point1 = nn.Parameter(torch.randn(1, 3))
        self.point2 = nn.Parameter(torch.randn(1, 3))
        self.point3 = nn.Parameter(torch.randn(1, 3))
        self.point4 = nn.Parameter(torch.randn(1, 3))
        self.q1 = nn.Parameter(torch.randn(1, 1))
        self.q2 = nn.Parameter(torch.randn(1, 1))
        self.q3 = nn.Parameter(torch.randn(1, 1))
        self.q4 = nn.Parameter(torch.randn(1, 1))
        self.linear = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
    
    def calculate_a(self, x):
        a1 = self.q1/(torch.norm(x - self.point1, dim=1, keepdim=True)**3) * (x - self.point1)
        a2 = self.q2/(torch.norm(x - self.point2, dim=1, keepdim=True)**3) * (x - self.point2)
        a3 = self.q3/(torch.norm(x - self.point3, dim=1, keepdim=True)**3) * (x - self.point3)
        a4 = self.q4/(torch.norm(x - self.point4, dim=1, keepdim=True)**3) * (x - self.point4)
        return a1 + a2 + a3 + a4
    def forward(self, x, time):
        k1 = self.linear(x)
        a1 = self.calculate_a(x)
        dx1 = self.h/2 * k1 + (self.h/2)**2 * a1/2
        k2 = self.linear(x + dx1)
        a2 = self.calculate_a(x + dx1)
        dx2 = self.h/2 * k2 + (self.h/2)**2 * a2/2
        k3 = self.linear(x + dx2)
        a3 = self.calculate_a(x + dx2)
        dx3 = self.h * k3 + self.h**2 * a3/2
        k4 = self.linear(x + dx3)
        return x + self.h/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def get_model(args):
    if args.model == 'lstm':
        return LSTM_Model(args)
    elif args.model == 'transformer':
        # transformer
        return Transformer_model(args)
    elif args.model == 'transformer_linear':
        return Transformer_model_linear(args)
    elif args.model == 'rk4':
        return RungeKutta_4(args)
    elif args.model == 'rk4Trival':
        return Trival_model(args)
    elif args.model == 'linear':
        return Polynomial_model(args)
