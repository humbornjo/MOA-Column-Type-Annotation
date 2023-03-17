import nltk
import torch
import json
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
import numpy as np
import pickle
import Levenshtein
from sklearn.metrics import confusion_matrix, f1_score

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

with open('../data/ssy_test.coltype.pkl', 'rb') as f:
    df = pickle.load(f)
with open('../data/mapping_dict.json', 'r', encoding='utf-8') as f:
    map_dict = json.load(f)
# print(df.loc[0]['type'])

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

fb = sorted(map_dict.items(), key = lambda x:x[1])
fb_sorted_list = [(i[0].split('.')[1]+' of '+i[0].split('.')[0]).replace('_', ' ') for i in fb]
tokenized_fb_types = tokenizer(fb_sorted_list, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    fb_embeddings = model(**tokenized_fb_types, output_hidden_states=True, return_dict=True).pooler_output

ts = 0
count = 0
alpha = 0.3
beta = 1-alpha
gamma = 0.1

pred_list = []
gd_list = []

# print("table_idx".ljust(10),"dict_len".ljust(10), 
#       "pred_idx".ljust(10), "curr_acc".ljust(10), "result")

for num, row in df.iterrows():

    if len(row['labels'])>1:
        continue
    # if count>5:
    #     break
    
    count+=1
    
    # if count<30:
    #     continue
    
    complex_list = sorted(row[-1].items(), key = lambda x:x[1], reverse=True)
    target_list = [i[0].replace('_', ' ') if 'http://dbpedia.org/ontology/' not in i[0] else i[0].split('/')[-1] for i in complex_list] 
    count_list = [i[1] for i in complex_list] 
    
    if not target_list:
        hd_ls = ls_cal([row['header']], fb_sorted_list)
        hd_es = es_cal(model, [row['header']], fb_embeddings)
        
        idx = np.argmax(softmax(hd_ls)*alpha + softmax(hd_es)*beta)

        pred_list.append(idx)
        gd_list.append(map_dict[row['labels'][0]])
        if fb[idx][0] in row['labels']:
            ts+=1
        
        # print(("%d" % num).ljust(10), 
        #         ("%d" % len(count_list)).ljust(10), 
        #         ("%d" % idx).ljust(10),          
        #         ("%.3f" % (ts/count)).ljust(10),
        #         [fb[idx][0]],' ->', row['labels'])
        continue
    
    ls = np.array(count_list).dot(Z_Score_Normalization(ls_cal(target_list, fb_sorted_list)))
    es = np.array(count_list).dot(Z_Score_Normalization(es_cal(model, target_list, fb_embeddings)))
    
    hd_ls = ls_cal([row['header']], fb_sorted_list)
    hd_es = es_cal(model, [row['header']], fb_embeddings)
    
    # print(es.shape,.shape, np.array(count_list).dot(es).shape) 
    idx = np.argmax(softmax(ls)*alpha + softmax(es)*beta + (softmax(hd_ls)*alpha + softmax(hd_es)*beta)*gamma
                    )

    pred_list.append(idx)
    gd_list.append(map_dict[row['labels'][0]])
    if fb[idx][0] in row['labels']:
        ts+=1
        
    # print(("%d" % num).ljust(10), 
    #       ("%d" % len(count_list)).ljust(10), 
    #       ("%d" % idx).ljust(10),          
    #       ("%.3f" % (ts/count)).ljust(10),
    #       [fb[idx][0]],' ->', row['labels'])
        
    # if count>=30:
    #     break
    
eval_dict = defaultdict(dict)
ts_micro_f1 = f1_score(gd_list,
                        pred_list,
                        average="micro")
ts_macro_f1 = f1_score(gd_list,
                        pred_list,
                        average="macro")
ts_class_f1 = f1_score(gd_list,
                        pred_list,
                        average=None,
                        labels=np.arange(255))
ts_conf_mat = confusion_matrix(gd_list,
                                pred_list,
                                labels=np.arange(255))
eval_dict["ts_micro_f1"] = ts_micro_f1
eval_dict["ts_macro_f1"] = ts_macro_f1
if type(ts_class_f1) != list:
    ts_class_f1 = ts_class_f1.tolist()
eval_dict["ts_class_f1"] = ts_class_f1
if type(ts_conf_mat) != list:
    ts_conf_mat = ts_conf_mat.tolist()
eval_dict["confusion_matrix"] = ts_conf_mat

output_filepath = "./kg_result.json"

with open(output_filepath, "w") as fout:
    json.dump(eval_dict, fout)