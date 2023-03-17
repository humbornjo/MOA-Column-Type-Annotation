from collections import defaultdict
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import confusion_matrix, f1_score

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from dataset import (collate_fn,collate_fn_ssy, TURLColTypeColwiseDataset,
                     TURLColTypeTablewiseDataset, TURLRelExtColwiseDataset,
                     TURLRelExtTablewiseDataset, SatoCVColwiseDataset,
                     SatoCVTablewiseDataset, TURLColTypeMultiModeTablewiseDataset)

from model import BertForMultiOutputClassification, BertMultiPairPooler
from util import parse_tagname, ssy_f1_score_multilabel, wiki_id2url, list2dataframe


## lib for kg module
import nltk
import torch
import json
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pickle
import Levenshtein

def get_dist_cosine(emb1, emb2):
    vector_a = np.array(emb1)
    vector_b = np.array(emb2)
    cos=vector_a.dot(vector_b.T)/(np.reshape(np.linalg.norm(vector_a,axis=1),(vector_a.shape[0],1))* np.linalg.norm(vector_b,axis=1))
    return np.where(np.isnan(cos),0,cos)

def es_cal(model, tar_list, comp_embeddings):
    token_id = tokenizer(tar_list, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        type_embeddings = model(**token_id, output_hidden_states=True, return_dict=True).pooler_output
    es = get_dist_cosine(type_embeddings, comp_embeddings)
        
    return es

def get_dist_leven(main_sent, sub_sent):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', None])
    mains = [i for i in nltk.word_tokenize(main_sent.lower()) if i not in stop_words]
    subs = [i for i in nltk.word_tokenize(sub_sent.lower()) if i not in stop_words]

    res_list = []

    for token in subs:
        sim = 0
        for text in mains:
            edist = Levenshtein.ratio(token, text)
            if edist > sim:
                sim = edist
        res_list.append(sim)
    error = len(subs) // 5
    while error > 0:
        res_list.pop(int(np.argmin(res_list)))
        error -= 1

    return np.mean(res_list)

def ls_cal(tar_list, comp_list):
    res_list = []
    for target in tar_list:
        temp_list = []
        for comp in comp_list:
            temp_list.append(get_dist_leven(target, comp))
        res_list.append(temp_list)
    ls = np.array(res_list)
            
    return ls

def Z_Score_Normalization(x):
    return (x - np.mean(x))/(np.std(x))

def softmax(f):
    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f) # f becomes [-666, -333, 0]
    return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer

    
if __name__ == '__main__':
    
    ## ml module prep
    task = "turl"
    batch_size = 16
    num_classes = 255
    tag_name = sys.argv[1]
    multicol_only = False
    shortcut_name, _, max_length = parse_tagname(tag_name)

    output_filepath = "{}={}_ssy.json".format(
        tag_name.replace("model/", "eval/"), task)
    f1_macro_model_path = "{}_best_macro_f1.pt".format(tag_name)
    f1_micro_model_path = "{}_best_micro_f1.pt".format(tag_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(shortcut_name)
    model = BertForMultiOutputClassification.from_pretrained(
                shortcut_name,
                num_labels=num_classes,
                output_attentions=False,
                output_hidden_states=False,
            ).to(device)
    
    
     
#########################################################################################
#########################################################################################
#########################################################################################
    
    # turl_dataset = json.load(open('/home/shensy/Code/python/doduo/data/turl_dataset/test.table_col_type.json',"rb"))
    # idx_dict = json.load(open('./mapping_dict.json',"r"))
    
    
    # id_set = set()

    # fineg_pre_list = []
    # for table in turl_dataset:
    #     [table_idx, page_title, table_id, headline, table_name, header, data, gd_header] = table
        
    #     table_id  = [table_idx for i in range(data)]
    #     list_col_data = [[] for i in range(len(data))]
    #     list_type_dict = [{} for i in range(len(data))]
    #     labels = gd_header
    #     header = header
        
    #     for i,row in enumerate(data):
    #         for cell in row:
    #             list_col_data[i].append(cell[-1][-1]) 
    #             try:
    #                 list_type_dict[i][cell[-1][0]]+=1
    #             except:
    #                 list_type_dict[i][cell[-1][0]]=1
        
    #     list_col_data = [' '.join(l) for l in list_col_data]
    #     label_ids = [np.zeros(len(idx_dict)) i in range(data)]
    #     for i, ids in enumerate(label_ids):
    #         ids[[idx_dict[t] for t in gd_header[i]]] = 1
    #     fineg_pre_list += zip(table_id,labels,list_col_data,label_ids,header,list_type_dict)

    # ssy_df = pd.DataFrame(fineg_pre_list,
    #                     columns=[
    #                         "table_id", "labels", "data",
    #                         "label_ids", "header", "type"
    #                     ])

#########################################################################################
#########################################################################################
#########################################################################################

    filepath = './data/fine-grained_WikiTables.pkl'
    dataset_cls = TURLColTypeMultiModeTablewiseDataset
    test_dataset = dataset_cls(filepath=filepath,
                                tokenizer=tokenizer,
                                max_length=max_length,
                                multicol_only=False,
                                device=device)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn_ssy)


    eval_dict = defaultdict(dict)
    for f1_name, model_path in [("f1_macro", f1_macro_model_path),
                                ("f1_micro", f1_micro_model_path)]:
        model.load_state_dict(torch.load(model_path, map_location=device))
        ts_pred_list = []
        ts_true_list = []
        ori_input_list = []
        
        
        # Test
        for batch_idx, batch in enumerate(test_dataloader):
            
 
            # Multi-column            
            logits, = model(batch["data"].T)
            if len(logits.shape) == 2:
                logits = logits.unsqueeze(0)
            cls_indexes = torch.nonzero(
                batch["data"].T == tokenizer.cls_token_id)
            filtered_logits = np.zeros((cls_indexes.shape[0],
                                            logits.shape[2]))
            # ori_input_list += batch["data"].cpu().detach().numpy(
            #     ).tolist()
            for n in range(cls_indexes.shape[0]):
    
                i, j = cls_indexes[n]
                logit_n = logits[i, j, :]
                logit_n = logit_n.detach().cpu().numpy()
                filtered_logits[n] = ( logit_n >= math.log(0.5))  #>= logit_n.max()
            ts_pred_list += (filtered_logits 
                                ).tolist()
            ts_true_list += batch["label"].cpu().detach(
            ).numpy().tolist()

            
            ### batch_size * according_col_number
            ### res_list = []
            ### for i in range(batch_size.length):
            ###     for dict in type_dict_tensor[i]:
            ###         do res_list.append(sim)
            ### add with filtered_logits 
            ### argmax or threshold
            
            
            
        print(ts_pred_list.__len__(),'\t', np.array(ts_pred_list).sum())


        # print(ori_input_list)

        ts_micro_f1, ts_macro_f1, ts_class_f1, ts_conf_mat = ssy_f1_score_multilabel(
                ts_true_list, ts_pred_list)

        eval_dict[f1_name]["ts_micro_f1"] = ts_micro_f1
        eval_dict[f1_name]["ts_macro_f1"] = ts_macro_f1
        if type(ts_class_f1) != list:
            ts_class_f1 = ts_class_f1.tolist()
        eval_dict[f1_name]["ts_class_f1"] = ts_class_f1
        if type(ts_conf_mat) != list:
            ts_conf_mat = ts_conf_mat.tolist()
        eval_dict[f1_name]["confusion_matrix"] = ts_conf_mat

    with open(output_filepath, "w") as fout:
        json.dump(eval_dict, fout)

