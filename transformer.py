import torch
import numpy as np
import pandas as pd
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

DEVICE = torch.device('cpu')

# 创建一个数据加载器，每次获取固定数量的batch，同时对batch进行补齐
class Loader:
  def __init__(self, data, batch_size):
    self.data = data
    self.batch_size = batch_size
    self.curr = 0
  def get_batch(self):
    #enlist = self.data.iloc[self.curr:self.curr+self.batch_size,[0]].values
    #print(list(map(float,enlist[0])))

    batch_en = torch.nn.utils.rnn.pad_sequence(
        [torch.Tensor(list(map(float,en[0]))) for en in self.data.iloc[self.curr:self.curr+self.batch_size,[0]].values],batch_first=True)
    batch_zh = torch.nn.utils.rnn.pad_sequence(
        [torch.Tensor(list(map(float,zh[0]))) for zh in self.data.iloc[self.curr:self.curr+self.batch_size,[1]].values],batch_first=True)
    return batch_en, batch_zh



class InputEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model):
    super(InputEmbedding, self).__init__()
    # 一个普通的 embedding 层, vocab_size词表长度， d_model每个单词的维度 
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.d_model = d_model

  def forward(self,x):
    x = self.embedding(x) * math.sqrt(self.d_model) # 这里x的尺寸为 batch_size(句子个数) * seq_len（每句单词个数） * d_model（单词维度）
    return x



class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=500):
    super(PositionalEncoding, self).__init__()

    # 初始化max_len×d_model的全零矩阵
    pe = torch.zeros(max_len, d_model, device=DEVICE)

    """
    unsqueeze作用：扩展维度
    x = torch.Tensor([1, 2, 3, 4]) #torch.Size([4]) 一维
    torch.unsqueeze(x, 0)) #tensor([[1., 2., 3., 4.]]) torch.Size([1, 4]) 一行四列，二维
    torch.unsqueeze(x, 1))
    tensor([[1.],
        [2.],
        [3.],
        [4.]]) #torch.Size([4, 1]) 四行一列
    """
    # 上面公式 P(i,2j) 与 P(i,2j+1) 中，i∈[0, max_len)表示单词在句子中的位置 2j∈[0, d_model)表示单词的维度
    position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1) # torch.Size([max_len, 1])
    # 两个公式中的 i/(10000^(2j/d_model)) 项是相同的，只需要计算一次即可
    # 这里幂运算太多，我们使用exp和log来转换实现公式中 i要除以的分母（由于是分母，要注意带负号）
    # torch.exp自然数e为底指数运算 与 math.log对数运算抵消
    div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))

    # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
    pe[:, 0::2] = torch.sin(position * div_term) #P(i,2j)=sin(...)，从0开始步长为2 表示2j
    pe[:, 1::2] = torch.cos(position * div_term) #P(i,2j+1)=cos(...)，从1开始步长为2 表示2j+1

    # 加1个batch维度，使得pe维度变为：1×max_len×d_model，方便后续与一个batch的句子所有词的embedding批量相加
    pe = pe.unsqueeze(0)
    # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
    self.register_buffer('pe', pe)

  def forward(self, x):
    # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加后输出，且尺寸不变
    # 设 x 的句子长度为len，尺寸为 batch×len×d_model, 这里x.size(1)即为句子长度len，则 self.pe[:, :x.size(1)]尺寸为 1×len×d_model
    x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
    return x





