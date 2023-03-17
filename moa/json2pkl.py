from SPARQLWrapper import SPARQLWrapper, JSON
import pickle as pkl

class DbSparql:
    def __init__(self,endpoint):
        self.endpoint=endpoint
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(3)

    def set_query(self,query):
        self.sparql.setQuery(query)
    def get_type(self,mention):
        if not mention:
            return []
        result_list = []  
        queryTypes = """
                SELECT DISTINCT str(?mtype) as ?mtype 
                WHERE {{
                    {{
                        ?val foaf:isPrimaryTopicOf <http://en.wikipedia.org/wiki/%s>.
                        ?val rdf:type ?mtype.
                    }}
                FILTER (strstarts(str(?mtype),'http://dbpedia.org/ontology/'))
                FILTER (strstarts(str(?val),'http://dbpedia.org/resource/'))
                }}
            """ % mention

        self.set_query(queryTypes)
        results = self.sparql.query().convert()

        for result in results["results"]["bindings"]:
            result_list.append(result["mtype"]["value"])

        if len(result_list) == 0:
            return []

        return result_list


class CaliSparql:

    def __init__(self,endpoint):
        self.endpoint=endpoint
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(3)

    def set_query(self,query):
        self.sparql.setQuery(query)


    def get_type(self,mention):
        if not mention:
            return []
        result_list = []
        queryTypes = """
                SELECT DISTINCT str(?mtype) as ?mtype 
                WHERE {{
                    {{
                        <http://caligraph.org/resource/%s> rdf:type ?mtype.
                    }}
                FILTER (strstarts(str(?mtype),'http://caligraph.org/ontology/'))
                }}
            """ % mention

        self.set_query(queryTypes)
        results = self.sparql.query().convert()


        for result in results["results"]["bindings"]:
            result_list.append(result["mtype"]["value"])

        if len(result_list) == 0:
            return []

        return result_list

        
        
        
        
import requests, json
import pandas as pd
import numpy as np

def wiki_id2url(pgid:int):   
    pgid= str(pgid)
    x = requests.get('https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids=%s&inprop=url&format=json' % pgid)
    return x.json()["query"]["pages"][pgid]["fullurl"]

def wiki_url2type(wikiurl:str, dbq, caliq):
    return dbq.get_type(wikiurl)+[ct.split('/')[-1] for ct in caliq.get_type(wikiurl)]

dbq=DbSparql(endpoint='https://dbpedia.org/sparql')
caliq=CaliSparql(endpoint='http://caligraph.org/sparql')

turl_dataset = json.load(open('/home/shensy/Code/python/doduo/data/turl_dataset/sgl_test_coltype - Copy.json',"r"))
idx_dict = json.load(open('./mapping_dict.json',"r"))

try:
    wiki_id2u = json.load(open('/home/shensy/Code/python/doduo/data/wiki_id2url.json',"r"))
except:
    wiki_id2u = {}
try:
    wiki2heter = json.load(open('/home/shensy/Code/python/doduo/data/wikiurl2type.json',"r"))
except:
    wiki2heter = {}

fineg_pre_list = []

count=0
for table in turl_dataset:
    count+=1
    print(count)
    [table_idx, page_title, table_id, headline, table_name, header, data, gd_header] = table
    
    table_id  = [table_idx for i in range(len(data))]
    list_col_data = [[] for i in range(len(data))]
    list_type_dict = [{} for i in range(len(data))]
    labels = gd_header
    header = header
    
    for i,row in enumerate(data):
        for j, cell in enumerate(row):
            if j > 20:
                break
            list_col_data[i].append(cell[-1][-1]) 
            if cell[-1][0] in wiki_id2u.keys():
                key = wiki_id2u[cell[-1][0]]
            else:
                try:
                    key = wiki_id2url(cell[-1][0]).split('/')[-1]
                except:
                    continue
                wiki_id2u[cell[-1][0]] = key
            if key in wiki2heter.keys():
                type_set = wiki2heter[key]
            else:
                try:
                    type_set = wiki_url2type(key,dbq,caliq)
                except:
                    continue
                wiki2heter[key] = type_set
            for t in type_set:
                try:
                    list_type_dict[i][t]+=1
                except:
                    list_type_dict[i][t]=1
    
    list_col_data = [' '.join(l) for l in list_col_data]
    label_ids = [np.zeros(len(idx_dict)) for i in range(len(data))]
    for i, ids in enumerate(label_ids):
        ids[[idx_dict[t] for t in gd_header[i]]] = 1
    fineg_pre_list += zip(table_id,labels,list_col_data,label_ids,header,list_type_dict)

ssy_df = pd.DataFrame(fineg_pre_list,
                    columns=[
                        "table_id", "labels", "data",
                        "label_ids", "header", "type"
                    ])

with open('/home/shensy/Code/python/doduo/data/wiki_id2url.json',"w") as f:
    json.dump(wiki_id2u,f)
with open('/home/shensy/Code/python/doduo/data/wikiurl2type.json',"w") as f:
    json.dump(wiki2heter,f)
with open('/home/shensy/Code/python/doduo/data/ssy_test.coltype.pkl',"wb") as f:
    pkl.dump(ssy_df,f)