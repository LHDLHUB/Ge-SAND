import os
import json
import copy
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertTokenizerFast, BertTokenizer, BertConfig
from GeSAND import GESAN, ELBCN
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
def split_bim_snp_name(bim_path):
    read = pd.read_csv(bim_path, sep="\t", header=None)
    snp_name = np.array(read[0])

    chrom_names = ["0"]
    snp_sort = ["0"]
    snp_num = 0
    for i in range(len(snp_name)):
        chrom_name = snp_name[i]
        if chrom_name in chrom_names:
            snp_num += 1
            snp_sort.append(snp_num)
        if chrom_name not in chrom_names:
            snp_num = 0
            snp_sort.append(snp_num)
        chrom_names.append(chrom_name)
    chrom_names = chrom_names + ["23"]
    snp_sort = snp_sort + ["0"]

    chrom_names = np.array(chrom_names).astype(int)
    snp_sort = np.array(snp_sort).astype(int)
    max_each_chrom_snp = np.max(snp_sort) + 1
    print(max_each_chrom_snp)
    torch.save(chrom_names, "chrom_names.ped")
    torch.save(snp_sort, "snp_sort.ped")
    return max_each_chrom_snp

class MyDataset(Dataset):
    def __init__(self, x, z, y):
        self.data_tensor = x
        self.data_z = z
        self.target_tensor = y

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.data_z[index], self.target_tensor[index]

def save_BERT_CSP(description_sentence, name):
    model_classify.eval()
    model.eval()
    model.to(device)

    description_sentence = np.array(description_sentence)
    predictions_1_CLS_all = []
    prediction_1_token_all_mean = []
    for i in range(0, len(description_sentence), 2):
        if i%5==0:
            print("i:", i)
            print(len(predictions_1_CLS_all))

        train_set = torch.tensor([]).cuda()
        segments_set = torch.tensor([]).cuda()


        for j in range(0, 2):

            text_1 = []
            text_1.append("[CLS]")
            text_1.append(description_sentence[i+j])
            text_1.append('[SEP]')

            text_1 = ' '.join(text_1)

            tokenized_text_1 = tokenizer.tokenize(text_1)


            indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)


            tokens_tensor_1 = torch.tensor([indexed_tokens_1])

            segments_ids = []

            text1_split = text_1.split(' ')

            while '' in text1_split:
                text1_split.remove('')

            for k in range(len(text1_split)):

                segments_ids.append(0)


            segments_tensors = torch.tensor([segments_ids]).to(device)
            tokens_tensor_1 = tokens_tensor_1.to(device)


            train_set = torch.cat([train_set, tokens_tensor_1], dim=0).cuda()
            segments_set = torch.cat([segments_set, segments_tensors], dim=0).cuda()

        with torch.no_grad():
            train_set = torch.tensor(train_set, dtype=torch.int).long().cuda()
            segments_set = torch.tensor(segments_set, dtype=torch.int).long().cuda()
            outputs_1 = model(train_set, token_type_ids=segments_set, return_dict=True,
                                        output_hidden_states=True)


        predictions_1 = outputs_1.hidden_states[-1]


        prediction_1_token = predictions_1[:, 1:-1, :]

        _, outputs= model_classify(prediction_1_token)  # 得到预测输出
        outputs_saved = outputs.detach().cpu().numpy().tolist()
        prediction_1_token_all_mean.append(outputs_saved)

    print(torch.tensor(prediction_1_token_all_mean).shape)
    prediction_1_token_all_mean = torch.tensor(prediction_1_token_all_mean).reshape(-1, config.hidden_size)
    print("prediction_1_token_all.shape", prediction_1_token_all_mean.shape)

    torch.save(prediction_1_token_all_mean.to(torch.device('cpu')), "weights_token_mean_%s.pth"%name)##Only change testing [UNK] can test the importance of the features


if __name__ == "__main__":

    torch.manual_seed(190807)  # Current CPU 190807

    torch.cuda.manual_seed(190807)  # Current GPU 190807

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 16

    config = BertConfig.from_json_file('./config.bin')
    #

    #################
    bim_path = "./final_5000_snp.bim"
    with open(bim_path, 'r') as f:
        line_count = sum(1 for line in f)
    max_chrom_snp = split_bim_snp_name(bim_path)


    config.max_position_embeddings = line_count + 2

    config.max_each_chrom_snp = max_chrom_snp

    model = GESAN(config)


    state_dict = torch.load('./GESAN_Best.pth', map_location='cuda:0')

    model.load_state_dict(state_dict)
    #
    tokenizer = BertTokenizer('./gene_vocab.txt', max_len=config.max_position_embeddings)

    model_classify = ELBCN(config.hidden_size, config.max_position_embeddings).to(device)
    model_classify.load_state_dict(torch.load('ELBCN_Best.pth', map_location='cuda:0'))

    ##########导入数据
    df = torch.load('./train_set_torch')
    description_sentence1 = df[0].tolist()
    description_sentence1 = [' '.join(map(str, x)) for x in description_sentence1]
    df = torch.load('./validate_set_torch')
    description_sentence2 = df[0].tolist()
    description_sentence2 = [' '.join(map(str, x)) for x in description_sentence2]
    df = torch.load('./test_set_torch')
    description_sentence3 = df[0].tolist()
    description_sentence3 = [' '.join(map(str, x)) for x in description_sentence3]
    accurracy_all_list = []

    ##########读取句子并输入到BERT，获得最后一层隐藏层输出的[CSP]数据，并保存

    description_sentence = [description_sentence1, description_sentence2, description_sentence3]
    names = ["training", "validating", "testing"]


    for i in range(3):
        save_BERT_CSP(description_sentence[i], names[i])