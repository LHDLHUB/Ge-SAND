import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertTokenizer, BertConfig
from numpy import *
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, accuracy_score, recall_score, precision_score, f1_score
from GeSAND import GESAN, FLBCN
import numpy as np
import os
import seaborn as sns
import networkx as nx
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



def accuracy_cal(output, target, output_auc, key):
    #print(output)
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
    #specificity = 100 * torch.sum(()&())

    if accuracy + recall == 0 :
        return 0
    f1 = 2 * (precision*recall)/(precision+recall)
    print("F1_{}: {:.2f}%".format(key, f1))


    if key == "Ge_SAND_figure_test_%s"%type or key == "Ge_SAND_figure_test" or "Ge_SAND_figure_validate_%s"%type or key == "Ge_SAND_figure_validate" :

        torch.save([target, output_auc], './%s_target_output_auc.pth'%key)

        if os.path.isdir('./%s/'%key) == False:
            os.makedirs('./%s/'%key)

        ##ROC
        fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output_auc.cpu().numpy())
        plt.figure()##
        plt.plot(fpr, tpr)
        plt.text(0.5, 0.5, 'AUC=%.3f' % AUC)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('./%s/ROC.png'%key)
        metrics = []
        for t in thresholds:
            pred = np.array([1 if x > t else 0 for x in output_auc.cpu().numpy()])
            #print(pred)
            acc = accuracy_score(target.cpu().numpy(), pred)
            recall = recall_score(target.cpu().numpy(), pred)
            precision = precision_score(target.cpu().numpy(), pred)
            F1_SCORE = f1_score(target.cpu().numpy(), pred)

            metrics.append([acc, recall, precision, F1_SCORE])

        metrics = np.array(metrics)
        acc_scores = metrics[:, 0]
        recall_scores = metrics[:, 1]
        precision_scores = metrics[:, 2]
        F1_SCORES = metrics[:, 3]

        best_acc_idx = np.argmax(acc_scores)
        #print(acc_scores[best_acc_idx])
        best_acc_threshold = thresholds[best_acc_idx]
        print("best_ACC", acc_scores[best_acc_idx], "\n", 'corresponding_Recall', recall_scores[best_acc_idx],
              "\n", 'corresponding_Precision', precision_scores[best_acc_idx], "\n", 'corresponding_F1_scores',
              F1_SCORES[best_acc_idx])
        print("best_ACC_idx_threshold", best_acc_threshold)

        best_recall_idx = np.argmax(recall_scores)
        best_recall_threshold = thresholds[best_recall_idx]
        print("best_Recall", recall_scores[best_recall_idx])
        print("best_Recall_idx_threshold", best_recall_threshold)

        best_precision_idx = np.argmax(precision_scores)
        best_precision_threshold = thresholds[best_precision_idx]
        print("best_precision", precision_scores[best_precision_idx])
        print("best_precision_idx_threshold", best_precision_threshold)

        best_F1_SCORES_idx = np.argmax(F1_SCORES)
        best_F1_SCORES_threshold = thresholds[best_F1_SCORES_idx]
        print("best_F1_SCORES", F1_SCORES[best_F1_SCORES_idx])
        print("best_F1_SCORES_idx_threshold", best_F1_SCORES_threshold)
        ###KS
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
        plt.savefig('./%s/KS.png'%key)
        print("~~KS~~", ks)

        ###PR
        precision, recall, _ = precision_recall_curve(target.cpu().numpy(), output_auc.cpu().numpy())
        ap = average_precision_score(target.cpu().numpy(), output_auc.cpu().numpy())
        plt.figure()#
        plt.plot(recall, precision)
        plt.text(0.5, 0.5, 'AP=%.3f' % ap)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve')
        plt.legend(loc="lower right")
        plt.savefig('./%s/PR.png'%key)
        print("~~PR~~", ap)

    return accuracy, recall, precision, f1, AUC

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data_tensor = x
        self.target_tensor = y

    
    def __len__(self):
        return len(self.data_tensor)


    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

type_list = ["30%"]
word_embedding_prob_list = [1 / 3]


max_word_length = 2189
for i in range(1):
    choose_set = "test"
    config = BertConfig.from_json_file('./config.bin')
    #
    batch_size = config.batch_size
    n = 0
    i = n
    type = type_list[i]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_ = './%s_set_torch'%choose_set

    df = torch.load(set_)
    description_sentence_test_ori = df[0].tolist()
    description_sentence_test = [' '.join(map(str, sublist)) for sublist in description_sentence_test_ori]
    test_predictions_1_CLS_all = np.array(description_sentence_test)

    test_labels = df[1]


    my_dataset_test = MyDataset(test_predictions_1_CLS_all, test_labels)

    tensor_dataloader_test = DataLoader(
        dataset=my_dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )


    test_output_get = torch.tensor([]).to(device)
    test_output_get_auc = torch.tensor([]).to(device)
    test_target_output_get = torch.tensor([]).to(device)

    bim_path = "./final_5000_snp.bim"
    with open(bim_path, 'r') as f:
        line_count = sum(1 for line in f)
    max_chrom_snp = split_bim_snp_name(bim_path)


    config.max_position_embeddings = line_count + 2

    config.max_each_chrom_snp = max_chrom_snp

    model_best = GESAN(config)


    state_dict = torch.load('./GESAN_Best_level1.pth', map_location='cuda:0')

    model_best.load_state_dict(state_dict)

    #
    tokenizer = BertTokenizer('./gene_vocab.txt', max_len=config.max_position_embeddings)

    #print(model_best)

    logistic_best = FLBCN(config.hidden_size).to(device)
    logistic_best.load_state_dict(torch.load('FLBCN_Best.pth', map_location="cuda:0"))


    ###############################################################################
    model_best.eval()
    logistic_best.eval()
    attn_scores_sum = torch.zeros(config.max_position_embeddings, config.max_position_embeddings).to(device)
    ################
    running_loss = 0
    batch_idx = 1
    criterion = torch.nn.BCELoss(reduction="mean")
    ###############

    with torch.no_grad():  # 
        for data in tensor_dataloader_test:  # 
            images, labels = data  # 
            test_set = torch.tensor([]).to(device)
            segments_set = torch.tensor([]).to(device)

            if torch.cuda.is_available():
                # inputs = inputs.cuda()
                labels = labels.to(device)

            for i in range(0, labels.shape[0]):

                inputs_temp = np.array(images[i])
                inputs_temp = inputs_temp.tolist()
                text_1 = []
                masked_token_list = []
                predicted_token_list = []
                text_1.append("[CLS]")
                text_1.append(inputs_temp)
                text_1.append('[SEP]')

                #print(text_1)

                text_1 = ' '.join(text_1)

                tokenized_text_1 = tokenizer.tokenize(text_1)

                indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)

                tokens_tensor_1 = torch.tensor([indexed_tokens_1])


                model_best.eval()

                model_best.to(device)
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


            attentions_query = []
            def hook_fn_query(module, input, output):
                #print(output.shape)
                attentions_query.append(output)

            attentions_key = []
            def hook_fn_key(module, input, output):
                attentions_key.append(output)



            attn_layer_query = model_best.bert.encoder.layer[0].attention.self.query

            attn_layer_key = model_best.bert.encoder.layer[0].attention.self.key

            handle_query = attn_layer_query.register_forward_hook(hook_fn_query)

            handle_key = attn_layer_key.register_forward_hook(hook_fn_key)




            ########
            test_set = torch.tensor(test_set, dtype=torch.int).long().to(device)
            segments_set = torch.tensor(segments_set, dtype=torch.int).long().to(device)
            #print(test_set)


            outputs_1 = model_best(test_set, token_type_ids=segments_set, return_dict=True,
                              output_hidden_states=True)



            ##############输出分数矩阵
            handle_query.remove()
            handle_key.remove()


            attn_matrix_query = attentions_query[0]
            attn_matrix_key = attentions_key[0]


            attn_scores = torch.matmul(attn_matrix_query, attn_matrix_key.transpose(2, 1))  # [11, 11]

            attn_scores = attn_scores/8

            attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)

            attn_scores_sum_temp = torch.sum(attn_scores, dim=0)


            attn_scores_sum += attn_scores_sum_temp



            predictions_1 = outputs_1.hidden_states[-1]

            prediction_1_token = predictions_1[:, 1:-1, :]

            outputs = logistic_best(prediction_1_token)  

            loss = criterion(outputs, labels)


            running_loss += loss.item()

            batch_idx += 1


            predicted = torch.round(outputs)

            test_output_get_auc = torch.cat((test_output_get_auc, outputs), dim=0)
            test_output_get = torch.cat((test_output_get, predicted), dim=0)
            test_target_output_get = torch.cat((test_target_output_get, labels), dim=0)
        print("Mean loss:", running_loss / batch_idx)
        accuracy_test, _, _, _, AUC = accuracy_cal(test_output_get, test_target_output_get, test_output_get_auc, key="Ge_SAND_figure_%s_%s"%(choose_set, type))

