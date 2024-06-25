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


model = Transformer(src_vocab, tgt_vocab, n_layers=6, d_model=512, n_head=8, DEVICE=DEVICE)
model.load_state_dict(torch.load('model-60-epoch.pth'))
model.eval()

def batch_greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    batch_size, src_seq_len = src.size()
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0

    print('src', src)
    memory = model.encode(src, src_mask)
    tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data).to(DEVICE)
    
    for s in range(max_len):
        tgt_mask = MyDataset.get_subsequent_mask(tgt)
        out = model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask))

        prob = model.generator(out[:, -1, :])
        pred = torch.argmax(prob, dim=-1)

        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)

        # print('tgt after', tgt)
        
        pred = pred.cpu().numpy()
        
        for i in range(batch_size):
            # print(stop_flag[i])
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break
    
    return results

def src_to_ids(text, device="cuda"):
    en_tokenizer = spm.SentencePieceProcessor()
    en_tokenizer.Load("en_tokenizer.model")
    
    ids = [torch.Tensor([en_tokenizer.bos_id()] + en_tokenizer.EncodeAsIds(text) + [en_tokenizer.eos_id()])]
    return torch.LongTensor(np.array(ids)).to(device)

# 打印模型翻译结果
def translate(text, model):
    with torch.no_grad():
        src = src_to_ids(text)
        src_mask = (src != 0).unsqueeze(-2).to(src.device)
        decode_result = batch_greedy_decode(model, src, src_mask)
        print(decode_result)
        zh_tokenizer = spm.SentencePieceProcessor()
        zh_tokenizer.Load("zh_tokenizer.model")
        

        print(zh_tokenizer.decode_ids(decode_result))
        # translation = [id_to_zhs[id] for id in decode_result]
        # print(translation)

translate("I have to go to sleep.", model)
translate("Mathematics is the part of science you could continue to do if you woke up tomorrow and discovered the universe was gone.", model)