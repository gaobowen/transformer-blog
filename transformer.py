import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

import sentencepiece as spm
from zhconv import convert


def Q2B(uchar):
  """判断一个unicode是否是全角数字"""
  if uchar >= u'\uff10' and uchar <= u'\uff19':
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)
  else:
    return uchar

def stringpartQ2B(ustring):
  return "".join([Q2B(uchar) for uchar in ustring])


def convertSimple(x):
  return stringpartQ2B(convert(x.values[0], 'zh-cn'))


all_data = pd.read_csv('en-zh.tsv',sep='\t',on_bad_lines='skip',names=['NO.1','en','NO.2','zh'])
# 繁体转简体
zh_data = all_data.iloc[:,[3]].apply(convertSimple, axis=1).rename('zhs',inplace=True)

all_data = pd.concat([all_data.iloc[:,[1]], zh_data], axis=1)

# 追加第二个数据集
data_b = pd.read_json('train.json')

all_data = pd.DataFrame({"en": np.append(all_data.iloc[:,[0]].values, data_b.iloc[:,[0]].values), 'zh': np.append(all_data.iloc[:,[1]].values, data_b.iloc[:,[1]].values)})

# 文本进行排序
all_data["Length"] = all_data['en'].str.len()

all_data.sort_values(by='Length', inplace=True)
# all_data = all_data.applymap(lambda x: x.lower()) #英文全部转为小写

print(type(all_data))
print(type(all_data.values))
all_data

# 创建一个数据加载器，每次获取固定数量的batch，同时对batch进行补齐
class MyDataset(Dataset):
    def __init__(self, data, device="cuda"):
        self.data = data
        
        self.zh_tokenizer = spm.SentencePieceProcessor()
        self.zh_tokenizer.Load("zh_tokenizer.model")

        self.en_tokenizer = spm.SentencePieceProcessor()
        self.en_tokenizer.Load("en_tokenizer.model")

        # 加入特殊字符
        self.PAD = self.en_tokenizer.pad_id()  # 0
        self.BOS = self.en_tokenizer.bos_id()  # 2
        self.EOS = self.en_tokenizer.eos_id()  # 3
        
        self.device = device

        self.en_data = [torch.Tensor([self.BOS] + self.en_tokenizer.EncodeAsIds(en[0]) + [self.EOS]) for en in self.data.iloc[:,[0]].values]
        self.zh_data = [torch.Tensor([self.BOS] + self.zh_tokenizer.EncodeAsIds(zh[0]) + [self.EOS]) for zh in self.data.iloc[:,[1]].values]

        assert len(self.en_data) == len(self.zh_data)

    def __getitem__(self, idx):
        en_ids = self.en_data[idx]
        zh_ids = self.zh_data[idx]
        return [en_ids, zh_ids]

    def __len__(self):
        return len(self.en_data)

    @staticmethod
    def get_pad_mask(seq, pad_idx):
        # 对 PAD 做屏蔽操作
        # batch * seqlen -> batch * 1 * seqlen
        return (seq != pad_idx).unsqueeze(-2)

    @staticmethod
    def get_subsequent_mask(seq):
        # decode 不能让后面单词的注意力影响到前面的单词
        batch_size, seq_len = seq.size()
        # torch.triu 保留上三角元素，diagonal=0保留对角线，diagonal=1不保留对角线
        # 保留下三角，包括对角线。因为 decoder输入的是完整句子，对 Q 做屏蔽，使得K看不到Q后面的单词
        # Attention = Q @ K^T = Q (seq_len_q x d) @ K^T (d x seq_len_k) = (seq_len_q x seq_len_k) 
        '''
            K(memory)
         -----------
         | [1, 0, 0]
        Q| [1, 1, 0]
         | [1, 1, 1]
        '''
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    def collate_fn(self, batch):
        en_ids = [torch.LongTensor(np.array(x[0])) for x in batch]
        zh_ids = [torch.LongTensor(np.array(x[1])) for x in batch]
        
        batch_en = torch.nn.utils.rnn.pad_sequence(en_ids, batch_first=True, padding_value=self.PAD)
        batch_zh = torch.nn.utils.rnn.pad_sequence(zh_ids, batch_first=True, padding_value=self.PAD)
        batch_en = batch_en.to(self.device)
        batch_zh = batch_zh.to(self.device)
        
        # 训练过程中采用 Teacher Forcing 模式, 将目标句子拆成 “<BOF>我吃午饭” 和 “我吃午饭<EOF>” 直接用目标输入去推理目标输出
        tgt_in = batch_zh[:,:-1]
        tgt_real = batch_zh[:,1:]
        src = batch_en

        # torch中的attention mask 尺寸 correct_3d_size = (bsz * num_heads, tgt_len, src_len)  => n k q
        # src 中要排除 PAD 对注意力的干扰
        src_mask = self.get_pad_mask(src, self.PAD)
        

        # decode 中要排除 PAD 和 后面单词 对注意力的干扰
        tgt_mask = self.get_pad_mask(tgt_in, self.PAD) & self.get_subsequent_mask(tgt_in)

        # torch.nn.MultiheadAttention 中的的mask True代表屏蔽，False表示通过
        # n_head = 4
        # src_mask = src_mask.repeat(n_head, 1, 1)
        # tgt_mask = tgt_mask.repeat(n_head, 1, 1)
        # src_mask = src_mask == False
        # tgt_mask = tgt_mask == False
        
        # print('src_mask', src_mask.shape, 'tgt_mask', tgt_mask.shape)
        return src, src_mask, tgt_in, tgt_real, tgt_mask
    

class InputEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model):
    super(InputEmbedding, self).__init__()
    # 一个普通的 embedding 层, vocab_size词表长度， d_model每个单词的维度 
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.d_model = d_model

  def forward(self,x):
    # 这里x的尺寸为 batch_size(句子个数) * seq_len（每句单词个数） * d_model（单词维度）
    x = self.embedding(x) * math.sqrt(self.d_model) 
    # print('self.embedding(x)', x)
    return x
  

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=500, device='cuda'):
    super(PositionalEncoding, self).__init__()

    # 初始化max_len×d_model的全零矩阵
    # print('max_len', max_len, 'd_model', d_model)
    pe = torch.zeros(max_len, d_model, device=device)

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
    position = torch.arange(0., max_len, device=device).unsqueeze(1) # torch.Size([max_len, 1])
    # 两个公式中的 i/(10000^(2j/d_model)) 项是相同的，只需要计算一次即可
    # 这里幂运算太多，我们使用exp和log来转换实现公式中 i要除以的分母（由于是分母，要注意带负号）
    # torch.exp自然数e为底指数运算 与 math.log对数运算抵消
    div_term = torch.exp(torch.arange(0., d_model, 2, device=device) * -(math.log(10000.0) / d_model))

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
  
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, attn_dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # shape = batch * head * len * d_k 
        # q k 计算点积，所以向量的维度 d_q 和 d_k 需要保证相等
        assert q.size(-1) == k.size(-1)

        d_k = k.size(-1)

        #计算注意力, 这里要计算单词间的关联度，最后两个维度必须保证是 len * d_k
        # attn = [batch * head * len * d_k] dot [batch * head * d_k * len]
        #      = batch * head * len = batch * head * attn_score
        attn = torch.matmul(q / math.sqrt(d_k), k.transpose(-2, -1))

        # 如果存在要进行mask的内容，则将注意力评分中需要屏蔽的部分替换成一个很大的负数
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # 对 attn_score 做归一化
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        output = torch.matmul(attn, v)

        return output, attn
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        # k 和 v 通过注意力权重进行交互，所以d_k和d_v可以有不同的维度
        self.d_k = d_k
        self.d_v = d_v 

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # 对 注意力输出 进行特征整理
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(dropout)
        # 最后的维度size 进行标准化，趋近于标准高斯分布 
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, attn_mask=None):
        mask = attn_mask
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # batch * len * d_model
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # 输入尺寸 batch x len x d_model
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: batch x head x len x d_k
        # 计算注意力要保证最后两个维度是 句子长度 * 单词维度
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # batch * len -> batch * 1 * len
            # 添加一个 head 维度 1，通过广播机制可以mask所有的head
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # 把维度 batch x head x len x d_k 还原成 batch x len x d_model, 方便做残差
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        
        q = self.layer_norm(q)

        return q, attn
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
    

class EncoderLayer(nn.Module):
    # d_model输入特征维度，d_hid 为 PositionwiseFeed 的隐藏层维度
    def __init__(self, d_model, d_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, x, src_mask=None):
        # seq_len = src_mask.shape[-1]
        # torch中的attention mask 尺寸 correct_3d_size = (bsz * num_heads, tgt_len, src_len)
        y, attn = self.self_attn(
            x, x, x, attn_mask=src_mask)
        y = self.ffn(y)
        return y, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, x, encoder_y, tgt_mask=None, cross_attn_mask=None):
        # print(x.shape, tgt_mask.shape)

        # decoder 自注意力
        # tgt_mask2 = torch.zeros((tgt_mask.shape[0], tgt_mask.shape[1], tgt_mask.shape[2]), device=tgt_mask.device).bool()
        
        decoder_y, decoder_attn = self.self_attn(x, x, x, attn_mask=tgt_mask)
        
        
        knowledge = encoder_y
        # 交叉注意力层
        # 这里的 decoder_y, encoder_y, encoder_y 理解成 Xq Xk Xv
        # 用 decoder 的 q 去 查询 encode 的 k-v 里的关联信息
        decoder_y, cross_attn = self.cross_attn(
            decoder_y, knowledge, knowledge, attn_mask=cross_attn_mask)
        
        decoder_y = self.ffn(decoder_y)
        return decoder_y, decoder_attn, cross_attn


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_head, hidden_scaler=4):
        super().__init__()

        assert d_model % n_head == 0
        # 512 / 8 = 64
        d_k = d_v = d_model //  n_head

        # 输入前 先标准化
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_model * hidden_scaler, n_head, d_k, d_v)
            for _ in range(n_layers)])
        

    def forward(self, src_vecs, src_mask):

        encoder_y = self.layer_norm(self.dropout(src_vecs))

        for enc_layer in self.layer_stack:
            encoder_y, enc_slf_attn = enc_layer(encoder_y, src_mask)

        return encoder_y


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_head, hidden_scaler=4):
        super().__init__()

        assert d_model % n_head == 0
        # 512 / 8 = 64
        d_k = d_v = d_model //  n_head

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_model * hidden_scaler, n_head, d_k, d_v)
            for _ in range(n_layers)])

    def forward(self, tgt_vecs, encoder_y, src_mask, tgt_mask):

        dec_output = self.layer_norm(self.dropout(tgt_vecs))
        
        # print(dec_output)
        # 交叉注意力不需要因果掩码，因为其目的是让解码器能够看到编码器的整个输出。但是，它通常需要借助源序列的填充掩码(src_mask)来忽略无关的填充令牌。
        # correct_3d_size = (bsz * num_heads, tgt_len, src_len)
        # cross_mask = torch.ones((tgt_mask.shape[0], tgt_mask.shape[-1], src_mask.shape[-1]), device=src_mask.device).bool() & src_mask
        cross_mask = src_mask.repeat(1, tgt_mask.shape[-1], 1)
        for dec_layer in self.layer_stack:
            dec_output, decoder_attn, cross_attn = dec_layer(
                dec_output, encoder_y, tgt_mask=tgt_mask, cross_attn_mask=cross_mask)
            
        # print(dec_output.shape)
        return dec_output


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # F.softmax(self.proj(x), dim=-1)  CrossEntropyLoss自带 softmax 功能, 这里不需要加
        # 思考实验：这里如果加上会出现什么现象？
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, n_layers, d_model, n_head, DEVICE="cuda"):
        super(Transformer, self).__init__()

        self.src_embed_pos = nn.Sequential(InputEmbedding(src_vocab, d_model), PositionalEncoding(d_model, device=DEVICE)).to(DEVICE)
        self.tgt_embed_pos = nn.Sequential(InputEmbedding(tgt_vocab, d_model), PositionalEncoding(d_model, device=DEVICE)).to(DEVICE)

        self.encoder = Encoder(n_layers, d_model, n_head).to(DEVICE)
        self.decoder = Decoder(n_layers, d_model, n_head).to(DEVICE)

        self.generator = Generator(d_model, tgt_vocab).to(DEVICE)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) #初始化权重


    def encode(self, src, src_mask):
        return self.encoder(self.src_embed_pos(src), src_mask)

    def decode(self, encoder_y, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed_pos(tgt), encoder_y, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的 k-v 参数传入，进行decode
        encoder_y = self.encode(src, src_mask)
        dec_y = self.decode(encoder_y, src_mask, tgt, tgt_mask)
        
        return self.generator(dec_y)

print("OK")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch import nn, optim
from torch.optim import Adam


DEVICE = 'cuda'

train_dataset = MyDataset(all_data, device=DEVICE)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32,
                                  collate_fn=train_dataset.collate_fn)

# test_dataset = MyDataset(all_data, device=DEVICE)
# test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32,
#                                   collate_fn=test_dataset.collate_fn)

src_vocab = train_dataset.en_tokenizer.GetPieceSize()
tgt_vocab = train_dataset.zh_tokenizer.GetPieceSize()

# 这里数据样本比较少,只用4层 256 4
model = Transformer(src_vocab, tgt_vocab, n_layers=6, d_model=512, n_head=8, DEVICE=DEVICE)

if True:
    model.load_state_dict(torch.load('model-60-epoch.pth'))

# optimizer parameter setting，越是小的数据集越需要小的 learning rate
init_lr = 1e-5
warmup = 1
epoch = 100
clip = 1.0
optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)

# 用于动态控制学习率的大小
# 在发现loss不再降低或者acc不再提高之后，降低学习率。
# factor触发条件后lr*=factor；
# patience不再减小（或增大）的累计次数；
patience = 7
factor = 0.9
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.PAD, reduction='sum', label_smoothing = 0.1)

def train_epoch(model, data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch in data:
        
        src, src_mask, tgt_in, tgt_real, tgt_mask = batch
        tgt_out = model(src, tgt_in, src_mask, tgt_mask)
        # 展平为 [batch_size * seq_length, vocab_size]
        output_reshape = tgt_out.contiguous().view(-1, tgt_out.shape[-1])
        
        # 展平为 [batch_size * seq_length]
        tgt_real = tgt_real.contiguous().view(-1)
        ntokens = (tgt_real != train_dataset.PAD).data.sum() #去掉 PAD 计算token平均损失
        # 损失函数可以逐个元素地计算交叉熵，可以类比成 “广播”
        loss = criterion(output_reshape, tgt_real) / ntokens
        loss.backward() # 反向
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() #更新权重
        optimizer.zero_grad() #梯度清零
        
        epoch_loss += loss.item() #直接获得对应的python数据类型
        if(math.isnan(epoch_loss)):  
            raise OSError(epoch_loss)
            
    return epoch_loss / len(data)

print('begin train')

for step in range(epoch+1):
    epoch_loss = train_epoch(model, train_dataloader, optimizer, criterion, clip)
    if step > warmup:
        scheduler.step(epoch_loss)
    print(f'{step} epoch_loss', epoch_loss)
    if epoch_loss < 1.5:
        torch.save(model.state_dict(), f'model-{step}.pth')

# torch.save(model.state_dict(), f'model-60-epoch.pth')
print('end train')




