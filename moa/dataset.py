from functools import reduce
import operator
import os
import pickle
import json
from pycm import ConfusionMatrix
from util import set_diag_zero

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from torch.nn.utils.rnn import pad_sequence


def collate_fn(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    # type = torch.nn.utils.rnn.pad_sequence(
    #     [sample["type"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    try:
        type = torch.nn.utils.rnn.pad_sequence(
        [sample["type"] for sample in samples])
        batch = {"data": data, "type":type, "label": label}
    except:
        batch = {"data": data, "label": label}

    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch


def collate_fn_ssy(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    # type = torch.nn.utils.rnn.pad_sequence(
    #     [sample["type"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    type = [sample["type"] for sample in samples]
    header = [sample["header"] for sample in samples]

    batch = {"data": data, "type":type, "label": label, "header": header}
    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch

class SatoCVColwiseDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "./data"):
        if device is None:
            device = torch.device('cpu')

        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        row_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue

            for _, row in group_df.iterrows():
                row_list.append(row)

        self.df = pd.DataFrame(row_list)

        # Convert into torch.Tensor
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(x,
                                 add_special_tokens=True,
                                 max_length=max_length + 2)).to(device))
        self.df["label_tensor"] = self.df["class_id"].apply(
            lambda x: torch.LongTensor([x]).to(device)
        )  # Can we reduce the size?

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }

## TODO: [[CLS]]+[AAAAAAAAA]+[,]+[have type [MASK]]+[[CLS]+...]     [CLS]+[MASK]=>ALL
## TODO: [[CLS]]+[AAAAAAAAA]+[,]+[have type [MASK]+[.]+[SEP]]+[BBBBBBBBB+...]     [CLS]=>[TYPE_A,...] [MASK]=>[A]

class SatoCVTablewiseDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "~/Code/python/doduo/data"):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"

        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))

                df_list.append(pd.read_csv(filepath))
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            ## if collect train dataset and reach the number required, break
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            # print(group_df["class_id"].values.shape)
            data_list.append(
                [index,
                 len(group_df), token_ids, group_df["class_id"].values.flatten().tolist(), class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "class_id", "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}

class SatoCVTablewiseDatasetForAnalysis(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "~/Code/python/doduo/data"):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"

        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))

                df_list.append(pd.read_csv(filepath))
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            ## if collect train dataset and reach the number required, break
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}



class TURLColTypeColwiseDataset(Dataset):
    """TURL column type prediction column-wise (single-column)"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        row_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            for _, row in group_df.iterrows():
                row_list.append(row)

        self.df = pd.DataFrame(row_list)
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(
                    x, add_special_tokens=True, max_length=max_length + 2)).to(
                        device)).tolist()

        self.df["label_tensor"] = self.df["label_ids"].apply(
            lambda x: torch.LongTensor([x]).to(device))

        if multicol_only:
            # Do nothing
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }


class TURLColTypeTablewiseDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["label_ids"].tolist()).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }


class TURLRelExtColwiseDataset(Dataset):
    """TURL column relation prediction column-wise (single-column)"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        row_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            group_df = group_df.sort_values("column_id")

            for j, (_, row) in enumerate(group_df.iterrows()):
                if j == 0:
                    continue

                row["data_tensor"] = torch.LongTensor(
                    tokenizer.encode(group_df.iloc[0]["data"],
                                     add_special_tokens=True,
                                     max_length=max_length + 2) +
                    tokenizer.encode(row["data"],
                                     add_special_tokens=True,
                                     max_length=max_length + 2)).to(device)

                row_list.append(row)

        self.df = pd.DataFrame(row_list)
        self.df["label_tensor"] = self.df["label_ids"].apply(
            lambda x: torch.LongTensor([x]).to(device))

        if multicol_only:
            # Do nothing
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}


class TURLRelExtTablewiseDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # It's probably already sorted but just in case.
            group_df = group_df.sort_values("column_id")

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["label_ids"].tolist()).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }


class LpnembedSatoCVTablewiseDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "~/Code/python/doduo/data"):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"

        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))

                ###############################
                test = os.getcwd()
                #########################

                df_list.append(pd.read_csv(filepath))
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            ## if collect train dataset and reach the number required, break
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue

            # TODO: There the sugar daddy comes, Idt lpn'embed would work
            # token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
            #     x, add_special_tokens=True, max_length=max_length + 2)).tolist(
            # )
            # token_ids = torch.LongTensor(reduce(operator.add,
            #                                     token_ids_list)).to(device)
            # cls_index_list = [0] + np.cumsum(
            #     np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            # for cls_index in cls_index_list:
            #     assert token_ids[
            #                cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            # cls_indexes = torch.LongTensor(cls_index_list).to(device)
            # class_ids = torch.LongTensor(
            #     group_df["class_id"].values).to(device)
            # data_list.append(
            #     [index,
            #      len(group_df), token_ids, class_ids, cls_indexes])

            # =====================================================================================
            # =====================================================================================

            # if b=='none':
            #     a=[" ".join(i.split(" ")[0:token_num1])  for i in a]
            #     a=" ".join(a)
            #     batch_tokenized_a = tokenizer(a, add_special_tokens=True,
            #                                     max_length=max_length, padding='max_length',truncation=True)
            #     input_ids_temp,token_type_ids_temp,attention_mask_temp=batch_tokenized_a['input_ids'],batch_tokenized_a['token_type_ids'],batch_tokenized_a['attention_mask']
            # else:

            #     a=[" ".join(i.split(" ")[0:token_num])  for i in a]
            #     b=[" ".join(i.split(" ")[0:token_num])  for i in b]

            #     batch_tokenized_a = tokenizer(a, add_special_tokens=False) 
            #     batch_tokenized_b = tokenizer(b, add_special_tokens=False) 


            #     input_ids_temp=[101]
            #     token_type_ids_temp=[0]
            #     attention_mask_temp=[1]


            #     for i in range(min(len(batch_tokenized_a['input_ids']),len(batch_tokenized_b['input_ids']))):
            #     input_ids_temp.extend(batch_tokenized_a['input_ids'][i])
            #     token_type_ids_temp.extend(batch_tokenized_a['token_type_ids'][i])
            #     attention_mask_temp.extend(batch_tokenized_a['attention_mask'][i])

            #     input_ids_temp.extend(batch_tokenized_b['input_ids'][i])
            #     token_type_ids_temp.extend(batch_tokenized_b['attention_mask'][i])
            #     attention_mask_temp.extend(batch_tokenized_b['attention_mask'][i])

            #     if len(input_ids_temp)>(max_length-1):
            #     input_ids_temp,token_type_ids_temp,attention_mask_temp=input_ids_temp[:max_length-1]+[102],token_type_ids_temp[:max_length-1]+[0],attention_mask_temp[:max_length-1]+[1]
            #     else:
            #     input_ids_temp,token_type_ids_temp,attention_mask_temp=input_ids_temp[:]+[102]+[0]*(max_length-len(input_ids_temp)-1),token_type_ids_temp[:]+[0]+[0]*(max_length-len(input_ids_temp)-1),attention_mask_temp[:]+[1]+[0]*(max_length-len(input_ids_temp)-1)
            
            if len(group_df)==1:
                token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                    x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
                token_ids = torch.LongTensor(token_ids_list[0]).to(device)
                cls_index_list = [0]
                for cls_index in cls_index_list:
                    assert token_ids[
                               cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                type_ids = torch.LongTensor([0] * len(token_ids)).to(device)
                class_ids = torch.LongTensor(
                    group_df["class_id"].values).to(device)
                # print(class_ids)
                data_list.append(
                    [index,
                        len(group_df), token_ids, type_ids, class_ids, cls_indexes])
            else:
                ## TODO: consider the first col as the primary key col and reterive the rest of the col in order 
                ## For a table with 4 data cols, the #generated data should be 3 (|col|-1) 
                # print(group_df["data"])
                token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                    x, add_special_tokens=False, max_length=max_length)).tolist(
                    )
                pkcol = token_ids_list[0]
                if not pkcol:
                    continue
        
                for tids in range(len(token_ids_list)-1):
                    batch_tokenized_a = token_ids_list[tids] 
                    batch_tokenized_b = token_ids_list[tids+1] 
                    if not batch_tokenized_b:
                        continue

                    cutoff = min(min(len(batch_tokenized_a),len(batch_tokenized_b)),max_length)
                    bounded_id_inputs = [list(item) for item in zip(batch_tokenized_a, batch_tokenized_b)]
                    # print(type(bounded_id_inputs[0]))
                    # print(batch_tokenized_a,batch_tokenized_b,'\n',list(bounded_id_inputs))
                    token_ids = list(reduce(operator.add, bounded_id_inputs, []))[:2*cutoff]#+[0]*(max_length-cutoff)*2
                    type_ids = [0,1]*cutoff
                    # print(token_ids,"==========================")
                    token_ids, type_ids = torch.LongTensor([101] + token_ids + [102]).to(device), torch.LongTensor([0] + type_ids + [0]).to(device)


                    cls_index_list = [0]
                    for cls_index in cls_index_list:
                        assert token_ids[
                                   cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                    cls_indexes = torch.LongTensor(cls_index_list).to(device)
                    class_ids = torch.LongTensor(
                        group_df["class_id"].values[tids:tids+1]).to(device)
                    data_list.append(
                        [index,
                         len(group_df), token_ids, type_ids, class_ids, cls_indexes])
                
                token_ids = torch.LongTensor([101] + token_ids_list[-1][:max_length] + [102]).to(device)
                cls_index_list = [0]
                for cls_index in cls_index_list:
                    assert token_ids[
                               cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                type_ids = torch.LongTensor([0] * len(token_ids)).to(device)

                sgl_class_ids=group_df["class_id"].values.tolist()[-1]
                sgl_class_ids = torch.LongTensor([sgl_class_ids]).to(device)
                data_list.append(
                    [index,
                        len(group_df), token_ids, type_ids, sgl_class_ids, cls_indexes])

                ## abandoned TODO: 读出表名 按照列名读取表中内容 替换group_df["data"]
                # table_name = group_df["table_id"].tolist()[0]
                # col_name_list = group_df["class"].tolist()
                # viznet_basedir = "~/Code/python/sato/table_data/viznet_tables"
                # temp_filepath = os.path.join(viznet_basedir,table_name)
                # temp_df = pd.read_csv(temp_filepath)
                # temp_df.columns = [coln.lower() for coln in temp_df.columns]
                # # print(temp_df,'\n',col_name_list)
                # target_data_list = []
                # explain_flag=False
                # for col_name in col_name_list:
                #     try:
                #         target_data_list.append(temp_df[col_name].tolist())
                #     except:
                #         try:
                #             for coln in temp_df.columns:
                #                 if col_name in coln:
                #                     target_data_list.append(temp_df[coln].tolist())
                #         except:
                #             explain_flag = True
                #             break
                # if explain_flag: continue
                # temp_input_sequence = ' '.join([str(temp) for temp in reduce(operator.add, zip(*target_data_list))
                #                                 if not pd.isnull(temp)])
                # token_ids = torch.LongTensor(tokenizer.encode(
                #     temp_input_sequence, add_special_tokens=True,
                #     max_length=len(col_name_list)*max_length+2)).to(device)

                # cls_index_list = [0]
                # for cls_index in cls_index_list:
                #     assert token_ids[
                #                cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                # cls_indexes = torch.LongTensor(cls_index_list).to(device)
                # class_ids = torch.LongTensor(
                #     group_df["class_id"].values[:1]).to(device)
                # data_list.append(
                #     [index,
                #      len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor", "type_tensor", 
                                         "label_tensor", "cls_indexes"
                                     ])

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "type": self.table_df.iloc[idx]["type_tensor"]
        }
        # "idx": torch.LongTensor([idx])}
        # "cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}


class LpnembedTURLColTypeColwiseDataset(Dataset):
    """TURL column type prediction column-wise (single-column)"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        row_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            if len(group_df)==1:
                token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                    x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
                token_ids = torch.LongTensor(token_ids_list[0]).to(device)
                cls_index_list = [0]
                for cls_index in cls_index_list:
                    assert token_ids[
                               cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                type_ids = torch.LongTensor([0] * len(token_ids)).to(device)
                class_ids = torch.LongTensor(
                    [group_df["label_ids"].values[0]]).to(device)
                # print(class_ids.shape, "Single")
                row_list.append(
                    [index,
                        len(group_df), token_ids, type_ids, class_ids, cls_indexes])
            else:
                ## TODO: consider the first col as the primary key col and reterive the rest of the col in order 
                ## For a table with 4 data cols, the #generated data should be 3 (|col|-1) 
                # print(group_df["data"])
                token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                    x, add_special_tokens=False, max_length=max_length)).tolist(
                    )
                pkcol = token_ids_list[0]
                if not pkcol:
                    continue
        
                for tids in range(len(token_ids_list)-1):
                    batch_tokenized_a = token_ids_list[tids] 
                    batch_tokenized_b = token_ids_list[tids+1] 
                    if not batch_tokenized_b:
                        continue

                    cutoff = min(min(len(batch_tokenized_a),len(batch_tokenized_b)),max_length)
                    bounded_id_inputs = [list(item) for item in zip(batch_tokenized_a, batch_tokenized_b)]
                    # print(type(bounded_id_inputs[0]))
                    # print(batch_tokenized_a,batch_tokenized_b,'\n',list(bounded_id_inputs))
                    token_ids = list(reduce(operator.add, bounded_id_inputs, []))[:2*cutoff]#+[0]*(max_length-cutoff)*2
                    type_ids = [0,1]*cutoff
                    # print(token_ids,"==========================")
                    token_ids, type_ids = torch.LongTensor([101] + token_ids + [102]).to(device), torch.LongTensor([0] + type_ids + [0]).to(device)


                    cls_index_list = [0]
                    for cls_index in cls_index_list:
                        assert token_ids[
                                   cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                    cls_indexes = torch.LongTensor(cls_index_list).to(device)
                    class_ids = torch.LongTensor(
                        [group_df["label_ids"].values[tids]]).to(device)
                    # print(class_ids.shape, "ABAB")
                    if type(class_ids)==tuple:
                        print(sgl_class_ids)
                    row_list.append(
                        [index,
                         len(group_df), token_ids, type_ids, class_ids, cls_indexes])
                
                token_ids = torch.LongTensor([101] + token_ids_list[-1][:max_length] + [102]).to(device)
                cls_index_list = [0]
                for cls_index in cls_index_list:
                    assert token_ids[
                               cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                type_ids = torch.LongTensor([0] * len(token_ids)).to(device)

                sgl_class_ids=group_df["label_ids"].values.tolist()[-1]
                sgl_class_ids = torch.LongTensor([sgl_class_ids]).to(device)
                if type(sgl_class_ids)==tuple:
                    print(sgl_class_ids)
                row_list.append(
                    [index,
                        len(group_df), token_ids, type_ids, sgl_class_ids, cls_indexes])


        self.df = pd.DataFrame(row_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor", "type_tensor", 
                                         "label_tensor", "cls_indexes"
                                     ])

        if multicol_only:
            # Do nothing
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "type": self.df.iloc[idx]["type_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }

class SatoCVColwiseDatasetForCl(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,  
            cm_analysis_dir: str,
            threshold: float = 0.15,
            num_data: int = 10000,
            device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))

        # self.df = df

        ################################ shensy start ####################################
        res_iterator = json.load(    #"../eval/sato4_mosato_bert_bert-base-uncased-bs16-ml-32__sato4-1.00=sato4.json"
            open(cm_analysis_dir,
                'r',encoding="utf8"))
        cm = res_iterator["f1_macro"]["confusion_matrix"]

        # get cast dict
        cm_obj = ConfusionMatrix(matrix = res_iterator["f1_macro"]["confusion_matrix"])
        mat_ = cm_obj.to_array()
        norm_ = mat_/mat_.sum(axis = 1)[:,None]
        set_diag_zero(norm_)
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
        # print(type(df["class_id"].values[0]),df["class_id"].values[0])
        df_iter = df.groupby("class_id")
        df_dict={}
        for group_df in df_iter:
            # print(group_df[0])
            class_id, df = group_df
            df_dict[class_id] = df
                
        for i, tt in enumerate(target_type):
            nts = cast_dict[tt]
            target_df = df_dict[tt]
            for i in range(int(choice_vec[i])):
                neg_df = pd.concat([df_dict[nt] for nt in nts], ignore_index=True)
                target_data = target_df.sample(n=2, replace = True)
                neg_data = neg_df.sample(n=6, replace = True)
                
                data.append([pad_sequence(
                                    target_data["data_tensor"].values.tolist()+
                                    neg_data["data_tensor"].values.tolist(),
                                    ).to(device), torch.zeros(1, dtype=torch.long).to(device)])
        self.data=data
        ####################################################################################

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx][0],
            "label": self.data[idx][1]
        }


class TURLColTypeMultiModeTablewiseDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            self.df =  pickle.load(fin)
        print("The number of row in df is %d" % len(self.df.index))
        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break
            
            if max([len(tl) for tl in group_df["labels"]])>1:
                continue
            
            # [WARNING] This potentially affects the evaluation results as well
            if len(group_df) > max_colnum:
                continue

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["label_ids"].tolist()).to(device)

            ## the feature of kg module
            class_type_dict = group_df["type"].tolist()
            class_headers = group_df["header"].tolist()
                

            data_list.append(
                [index,
                 len(group_df), token_ids, 
                 class_ids, cls_indexes, 
                 class_type_dict, class_headers])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "type", "header"
                                     ])
        print("The number of row in final_df is %d" % len(self.table_df.index))

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "type": self.table_df.iloc[idx]["type"],
            "header": self.table_df.iloc[idx]["header"]
        }
