from pycm import *
import csv, json, torch, matplotlib
import pandas as pd
from turtle import color
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
from dataset import SatoCVColwiseDatasetForCl
from transformers import BertTokenizer, BertConfig


def skip_diag_masking(A): 
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1) 

def set_diag_zero(A):
    A[np.diag_indices_from(A)]=0


# csv_iterator=csv.reader(open("./sato_cv_4.csv",'r',encoding="utf8"))

# type2num=dict()
# for line in csv_iterator:
#     try:
#         type2num[line[3]]=line[2]
#     except:
#         continue
# print(type2num)

x_data=[]
y_data=[]
x_num=[]
res_iterator=json.load(open("../eval/sato4_mosato_bert_bert-base-uncased-bs16-ml-32__sato4-1.00=sato4.json",'r',encoding="utf8"))
# for i in range(len(res_iterator["f1_macro"]["ts_class_f1"])):
#     if res_iterator["f1_macro"]["ts_class_f1"][i]<0.7:
#         y_data.append(res_iterator["f1_macro"]["ts_class_f1"][i])
#         x_data.append(type2num[str(i)])
#         x_num.append(i)
#     print(i, res_iterator["f1_macro"]["ts_class_f1"][i])

# res=[]
# for i in x_num:
#     print(sum(res_iterator["f1_macro"]["confusion_matrix"][i]),type2num[str(i)])
#     res.append(res_iterator["f1_macro"]["confusion_matrix"][i])
# dat = pd.DataFrame(data=res) 

# dat.to_csv("bad_acc_sato.csv")








##### threshold: FN+FP more than *% * 0.01
##### cm: directry of confusion matrix
##### num_data: number of generated data
res_iterator = json.load(
    open("../eval/sato4_mosato_bert_bert-base-uncased-bs16-ml-32__sato4-1.00=sato4.json",
         'r',encoding="utf8"))
cm = res_iterator["f1_macro"]["confusion_matrix"]
threshold = 0.15
num_data = 10000

# get cast dict
cm_obj = ConfusionMatrix(matrix = res_iterator["f1_macro"]["confusion_matrix"])
mat_ = cm_obj.to_array()
norm_ = mat_/mat_.sum(axis = 1)[:,None]
set_diag_zero(norm_)
# plt.figure(figsize=(100,100), dpi=300)
# cm_obj.plot(cmap = plt.cm.Reds, normalized = True, number_label= True, plot_lib = "matplotlib")
# # plt.show()
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(50, 50)
# fig.savefig('test2png.png', dpi=100)
target_type = np.argwhere(norm_.sum(1)>threshold).squeeze()
neg_type_mat = np.argsort(-norm_, axis=1)

cast_dict={}
for tt in target_type:
    temp_th = 0  ## temp threshold
    temp_nts = []  ## temp negative types
    for nt in neg_type_mat[tt]:
        temp_th += norm_[tt][nt]
        temp_nts.append(nt)
        if temp_th >= threshold:
            break
    cast_dict[tt] = temp_nts

# build dataset
# TODO: change to softmax, use factor tau to control the ratio  
data=[]
choice_vec = mat_[target_type].sum(1)/mat_[target_type].sum()*num_data

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = SatoCVColwiseDatasetForCl(cv='4',
                                    split="train",
                                    train_ratio=0.1,
                                    tokenizer=tokenizer,
                                    max_length=32)


print(choice_vec, len(target_type))


df_iter = train_dataset.df.groupby("class_id")
print(len(df_iter))

df_dict={}
for group_df in df_iter:
    class_id, df = group_df
    df_dict[class_id] = df
        
for i, tt in enumerate(target_type):
    nts = cast_dict[tt]
    target_df = df_dict[tt]
    for i in range(int(choice_vec[i])):
        neg_df = pd.concat([df_dict[nt] for nt in nts], ignore_index=True)
        target_data = target_df.sample(n=2, replace = True)
        neg_data = neg_df.sample(n=6, replace = True)
        
        data.append(target_data["data_tensor"].values.tolist()+neg_data["data_tensor"].values.tolist())

