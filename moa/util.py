import random
import requests
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import torch
import pandas as pd
from torch.nn import CosineSimilarity, LogSoftmax, Softmax


def skip_diag_masking(A): 
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1) 

def set_diag_zero(A):
    A[np.diag_indices_from(A)]=0


def cl_loss(embed_output, temperature = 1):
    cosim = CosineSimilarity(dim=2)
    target = embed_output[:,:1]
    samples = embed_output[:,1:]
    
    exp_output = (cosim(target, samples)/temperature).exp()
    return torch.mean(-torch.log(exp_output[:,0]/torch.sum(exp_output,dim=1)))

def cl_loss_ce(embed_output,labels, temperature = 1):
    cec=torch.nn.CrossEntropyLoss()
    
    cosim = CosineSimilarity(dim=2)
    target = embed_output[:,:1]
    samples = embed_output[:,1:]
    return cec(cosim(target, samples)/temperature, labels)
    

def f1_score_multilabel(true_list, pred_list):
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()
    print("#################\n",p,r)

    
    micro_f1 = 2 * p * r / (p  + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] /  conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] /  conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1.mean()
    return (micro_f1, macro_f1, class_f1, conf_mat)


def parse_tagname(tag_name):
    """sato_bert_bert-base-uncased-bs16-ml-256"""
    if "__" in tag_name:
        # Removetraining ratio
        tag_name = tag_name.split("__")[0]
    tokens = tag_name.split("_")[-1].split("-")
    shortcut_name = "-".join(tokens[:-3])
    max_length = int(tokens[-1])
    batch_size = int(tokens[-3].replace("bs", ""))
    return shortcut_name, batch_size, max_length


def set_seed(seed:int):
    """https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L58-L63"""    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """Add the following 2 lines
    https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    """
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    """
    For detailed discussion on the reproduciability on multiple GPU
    https://discuss.pytorch.org/t/reproducibility-over-multigpus-is-impossible-until-randomness-of-threads-is-controled-and-yet/47079
    """

def wiki_id2url(pgid:int):
    
    pgid= str(pgid)
    x = requests.get('https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids=%s&inprop=url&format=json' % pgid)
    return x.json()["query"]["pages"][pgid]["fullurl"]

def list2dataframe(list_table):
    table_idx, page_title, table_id, headline, table_name, header, data, gd_header = list_table
    table_idx=table_idx.split('-')[-1]
    list_data=[]
    for row in data:
        temp_row=[]
        for cell in row:
            temp_row.append(cell[-1][-1]) 
        list_data.append(temp_row)

    list_data = map(list, zip(*list_data))
    df=pd.DataFrame(list_data, columns = header)
    return table_idx, page_title, table_id, headline, table_name, header, df, gd_header

def ssy_f1_score_multilabel(true_list, pred_list):
    
    temp=np.array(true_list).sum(0)
    # print(np.where(temp>0)[0].shape, np.array(true_list).shape)
    
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()
    print("#################\n",p,r)
    
    micro_f1 = 2 * p * r / (p  + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] /  conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] /  conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1[np.where(temp>0)[0]].mean()#
    return (micro_f1, macro_f1, class_f1, conf_mat)

# if __name__ == "__main__":
#     input = torch.rand((8,8,768))
    
#     ocl=cl_loss(input)
#     print(ocl)
#     cel=cl_loss_ce(input)
#     print(cel)