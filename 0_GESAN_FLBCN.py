import os
import json
import copy
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertTokenizerFast, BertTokenizer, BertConfig
from GeSAND import GESAN, FLBCN
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def split_bim_snp_name(bim_path):
    read = pd.read_csv(bim_path, sep="\t", header=None)
    print(read)
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

def accuracy_cal(output, target, output_auc, key):
    correct = (output == target).sum().item()
    total = target.size(0)
    accuracy = 100 * correct / total
    print("Accuracy_{}: {:.2f}%".format(key, accuracy))

    true_positives = torch.sum((target==1)&(output==1))###TP
    false_negatives = torch.sum ((target==1)&(output==0))#####FN
    recall = 100 * true_positives/(true_positives+false_negatives)####TP/(TP+FN))
    print("Recall_{}: {:.2f}%".format(key, recall))

    precision = 100 * torch.sum((target==1)&(output==1))/(torch.sum((target==1)&(output==1))+torch.sum((target==0)&(output==1)))####TP/(TP+FP)

    print("Precision_{}: {:.2f}%".format(key, precision))

    AUC = roc_auc_score(target.cpu().numpy(), output_auc.cpu().numpy())
    print("AUC{}: {:.2f}%".format(key, AUC*100))


    if accuracy + recall == 0 :
        return 0
    f1 = 2 * (precision*recall)/(precision+recall)
    print("F1_{}: {:.2f}%".format(key, f1))

    if key == "test":

        ##ROC曲线
        fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output_auc.cpu().numpy())
        plt.figure()##保证图与图之间分开
        plt.plot(fpr, tpr)
        plt.text(0.5, 0.5, 'AUC=%.3f' % AUC)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')
        ###PR曲线
        precision, recall, thresholds = precision_recall_curve(target.cpu().numpy(), output_auc.cpu().numpy())
        ap = average_precision_score(target.cpu().numpy(), output_auc.cpu().numpy())
        plt.figure()##保证图与图之间分开
        plt.plot(recall, precision)
        plt.text(0.5, 0.5, 'AP=%.3f' % ap)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve')
        plt.legend(loc="lower right")
        plt.savefig('PR.png')
        ###KS曲线
        ks = np.max(tpr - fpr)
        plt.figure()
        plt.plot(thresholds, tpr, label='TPR')
        plt.plot(thresholds, fpr, label='FPR')
        plt.plot(thresholds, tpr - fpr, label='KS')
        plt.text(0.5, 0.5, 'KS=%.3f' % ks)
        plt.xlabel('Threshold')
        plt.ylabel('Value')
        plt.title('KS Curve')
        plt.legend(loc="lower right")
        plt.savefig('KS.png')


    return accuracy, recall, precision, f1, AUC

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data_tensor = x
        self.target_tensor = y

    # 返回数据集大小
    def __len__(self):
        return len(self.data_tensor)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

def train(epoch):
    model.train()
    model_classify.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(tensor_dataloader_train, 0):  #
        inputs, target = data

        train_set = torch.tensor([]).to(device)
        segments_set = torch.tensor([]).to(device)

        if torch.cuda.is_available():
            target = target.to(device)

        for i in range(0, target.shape[0]):
            inputs_temp = np.array(inputs[i])
            inputs_temp = inputs_temp.tolist()
            text_1 = []
            text_1.append("[CLS]")
            text_1.append(inputs_temp)
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


            train_set = torch.cat([train_set, tokens_tensor_1], dim=0).to(device)
            segments_set = torch.cat([segments_set, segments_tensors], dim=0).to(device)



        train_set = torch.tensor(train_set, dtype=torch.int).long().to(device)
        segments_set = torch.tensor(segments_set, dtype=torch.int).long().to(device)
        outputs_1 = model(train_set, token_type_ids=segments_set, return_dict=True,
                          output_hidden_states=True)

        predictions_1 = outputs_1.hidden_states[-1]

        prediction_1_token = predictions_1[:, 1:-1, :]

        outputs = model_classify(prediction_1_token)


        optimizer_FLBCN.zero_grad()
        optimizer_GESAN.zero_grad()

        loss = criterion(outputs, target)

        loss.backward()
        optimizer_FLBCN.step()
        optimizer_GESAN.step()


        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))

def validate(AUC_get, epoch, running_loss_max):

    model.eval()
    model_classify.eval()
    test_output_get = torch.tensor([]).to(device)
    test_output_get_auc = torch.tensor([]).to(device)
    test_target_output_get = torch.tensor([]).to(device)
    with torch.no_grad():  # 
        running_loss = 0.0
        batch_idx = 0
        for data in tensor_dataloader_validate:  #
            images, labels = data  #
            test_set = torch.tensor([]).to(device)
            segments_set = torch.tensor([]).to(device)

            if torch.cuda.is_available():
                labels = labels.to(device)

            for i in range(0, labels.shape[0]):

                inputs_temp = np.array(images[i])
                inputs_temp = inputs_temp.tolist()

                text_1 = []
                text_1.append("[CLS]")
                text_1.append(inputs_temp)
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

                test_set = torch.cat([test_set, tokens_tensor_1], dim=0).to(device)
                segments_set = torch.cat([segments_set, segments_tensors], dim=0).to(device)


            test_set = torch.tensor(test_set, dtype=torch.int).long().to(device)
            segments_set = torch.tensor(segments_set, dtype=torch.int).long().to(device)
            outputs_1 = model(test_set, token_type_ids=segments_set, return_dict=True,
                              output_hidden_states=True)


            predictions_1 = outputs_1.hidden_states[-1]

            prediction_1_token = predictions_1[:, 1:-1, :]

            outputs = model_classify(prediction_1_token)  #
            predicted = torch.round(outputs)

            test_output_get_auc = torch.cat((test_output_get_auc, outputs), dim=0)
            test_output_get = torch.cat((test_output_get, predicted), dim=0)
            test_target_output_get = torch.cat((test_target_output_get, labels), dim=0)
            loss = criterion(outputs, labels)


            running_loss += loss.item()
            if batch_idx % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))
            batch_idx += 1

        accuracy_test, Recall, precision, F1_SCORE, AUC = accuracy_cal(test_output_get, test_target_output_get, test_output_get_auc, key="validate")


    if AUC_get < AUC:
        AUC_get = AUC
        running_loss_max = running_loss
        torch.save(model_classify.state_dict(), 'FLBCN_Best.pth')
        torch.save(model.state_dict(), 'GESAN_Best_level1.pth')
        print("saved~~~~~~")

    return AUC_get, running_loss_max



def train_and_test(AUC_get, running_loss_max):
    for epoch in range(28):
        #scheduler.step()
        #scheduler_BERT.step()

        train(epoch)
        AUC_get, running_loss_max = validate(AUC_get, epoch, running_loss_max)


if __name__ == '__main__':
    torch.manual_seed(190807)  # Current CPU 190807

    torch.cuda.manual_seed(190807)  # Current GPU 190807

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
######Create the position files
    bim_path = "./xxxxx.bim"
    with open(bim_path, 'r') as f:
        line_count = sum(1 for line in f)

    max_chrom_snp = split_bim_snp_name(bim_path)

######Build the model

    config = BertConfig.from_json_file('./config.bin')
    #
    batch_size = config.batch_size

    config.max_position_embeddings = line_count + 2

    config.max_each_chrom_snp = max_chrom_snp


    lr1 = config.lr_GESAN  ##
    lr2 = config.lr_BCN
    model = GESAN(config)

    model.to(device)

    #
    tokenizer = BertTokenizer('./gene_vocab.txt', max_len=config.max_position_embeddings)
    #

    train_set = torch.load("./train_set_torch")
    description_sentence_train_ori = train_set[0].tolist()
    description_sentence_train = [' '.join(map(str, sublist)) for sublist in description_sentence_train_ori]

    validate_set = torch.load("./validate_set_torch")
    description_sentence_validate_ori = validate_set[0].tolist()
    description_sentence_validate = [' '.join(map(str, sublist)) for sublist in description_sentence_validate_ori]

    test_set = torch.load("./test_set_torch")
    description_sentence_test_ori = test_set[0].tolist()
    description_sentence_test = [' '.join(map(str, sublist)) for sublist in description_sentence_test_ori]

    accurracy_all_list = []

    train_predictions_1_CLS_all = np.array(description_sentence_train)

    train_labels = train_set[1]

    validate_predictions_1_CLS_all = np.array(description_sentence_validate)

    validate_labels = validate_set[1]

    test_predictions_1_CLS_all = np.array(description_sentence_test)

    test_labels = test_set[1]

    my_dataset_train = MyDataset(train_predictions_1_CLS_all, train_labels)

    my_dataset_validate = MyDataset(validate_predictions_1_CLS_all, validate_labels)

    my_dataset_test = MyDataset(test_predictions_1_CLS_all, test_labels)

    tensor_dataloader_train = DataLoader(
        dataset=my_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    # print('tensor_dataloader_train:', tensor_dataloader_train)

    tensor_dataloader_validate = DataLoader(
        dataset=my_dataset_validate,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    tensor_dataloader_test = DataLoader(
        dataset=my_dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    model_classify = FLBCN(config.hidden_size).to(device)

    criterion = torch.nn.BCELoss(reduction="mean")

    if torch.cuda.is_available():
        model_classify = model_classify.to(device)
        criterion = criterion.to(device)

    #
    optimizer_GESAN = optim.AdamW(model.parameters(), lr=lr1)
    optimizer_FLBCN = optim.AdamW(model_classify.parameters(), lr=lr2)  # #

    AUC_get = 0
    running_loss_max = 1000000
    train_and_test(AUC_get, running_loss_max)

    ######################

    #cluster_model()




