import torch
import torch.nn as nn
import math
def generate_square_subsequent_mask( sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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
        self.positional_encoding = PositionalEncoding(args.input_size)
    def forward(self, x, label):
        batch_size = x.size(0)        
        if label == None:
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
        self.transformer = nn.Transformer(
            d_model=args.input_size,
            nhead=1,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1
        )
        self.linear = nn.Linear(args.input_size, 3)
        self.fc = nn.Linear(args.length, args.window)  
        self.output_len = args.window
    def forward(self, x):
        output = self.transformer(x, x)
        output = output.permute(0, 2, 1)
        output = self.fc(output)
        output = output.permute(0, 2, 1)
        output = self.linear(output)
        return output
def get_model(args):
    if args.model == 'lstm':
        return Model(args)
    elif args.model == 'transformer':
        # transformer
        return Transformer_model(args)
    elif args.model == 'transformer_linear':
        return Transformer_model_linear(args)
