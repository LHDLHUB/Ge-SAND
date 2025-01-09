# 导入依赖
import torch
import torch.nn as nn
from GenomicEmbedding import GenomicEmbedding
from transformers import BertModel, BertForMaskedLM, BertTokenizerFast, BertTokenizer, BertConfig
class GESAN(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.bert.embeddings = GenomicEmbedding(config)  # 使用自定义的嵌入层


class FLBCN(torch.nn.Module):
    def __init__(self, dense_size):
        super(FLBCN, self).__init__()

       ##model2
        self.dense_size = dense_size
        self.l1 = torch.nn.Linear(self.dense_size, 1)###
        self.sigmoid = torch.nn.Sigmoid()
        self.LN1D1 = torch.nn.LayerNorm(self.dense_size)
        self.dropout = torch.nn.Dropout(0.1)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)

    def forward(self, x):

        x = torch.mean(x, dim=1)
        x = x.view(-1, self.dense_size)
        x = self.LN1D1(x)
        x = self.l1(x)
        x = 0.5 * (torch.tanh(x) + 1)##best
        x = x.reshape(-1)
        return x

class ELBCN(torch.nn.Module):
    def __init__(self, dense_size, output_number):
        super(ELBCN, self).__init__()
        self.dense_size = dense_size
        self.l1 = torch.nn.Linear(self.dense_size, 1)###
        self.sigmoid = torch.nn.Sigmoid()
        self.LN1D1 = torch.nn.LayerNorm(self.dense_size)
        self.weight = torch.nn.Parameter(torch.empty(1, output_number-2).fill_(1/(output_number-2)))###这一步必须有，你要改就改L2勾函数；还有增强泛化能力BN or L2
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)

    def forward(self, x):
        x = self.weight @ x
        x = x.view(-1, self.dense_size)
        layernorm = self.LN1D1(x)
        x = self.l1(layernorm)
        x = 0.5 * (torch.tanh(x) + 1)##best
        x = x.reshape(-1)
        return x, layernorm

    def normalize(self):
       weight_copy = self.weight.clone()
       sum = torch.sum(weight_copy)
       weight_copy.div_(sum)
       self.weight.data = weight_copy.data
       self.weight.requires_grad = True