import tika
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
from tika import parser
from sklearn.cluster import KMeans
tika.initVM()

def extract_emails_content(data_path):
    """
    save emails content
    """
    with open(data_path, 'r') as data_file:
        data = json.load(data_file)
    out = {}
    i = 0
    for d in data:
        
        if "X-TIKA:content" in d.keys():
            out[i] = d["X-TIKA:content"]
        else:
            out[i] = ""
        i += 1
    return out
def is_tag(tag):
    """
    check if tag belongs to tag
    """
    return ord(tag) >= 32 and ord(tag) <= 47 or ord(tag) >= 58 and ord(tag) <= 64 or ord(tag) >= 91 and ord(tag) <= 96 or ord(tag) >= 123 and ord(tag) <= 126

def calculate_ttr_score(text):
    """
    calculate ttr score 
    """
    tags = 0
    non_tags = 0
    
    for t in text:
        if is_tag(t):
            tags+=1
        else:
            non_tags+=1
    if tags == 0:
        return non_tags
    return non_tags/tags

def process_corpus_t2t(corpus):
    """
    process corpus t2t
    return comment to TTR score
    """
    corpus_text_list = corpus.splitlines()
    res = {}
    for c in corpus_text_list:
        # remove scripts tag and remark tags
        if len(c) > 0 and c[0] == '<':
            res[c] = 0
            continue
        ttr_score = calculate_ttr_score(c)
        res[c] = ttr_score

    return res

def text2tag(data_path):
    """
    text to tag algorithm
    - Remove empty lines and script tags
    - Initialize the TTRArray
    - For each line in document
    ● X = number of non tag ASCII characters
    ● Y = numbers of tags in the line
    ● TTRArray[current line] = X if no tags, else TTRArray[current line] = X / Y
    """
    ttrs_array = {}

    corpus_data = extract_emails_content(data_path)
    # calculate TTR 
    for k in corpus_data.keys():
        print(k)
        ttrs_array[k] = process_corpus_t2t(corpus_data[k])
    
    out_file = open("emails_ttr_scores.json", "w") 
    json.dump(ttrs_array, out_file)
    out_file.close()
    
    return ttrs_array

def cluster_based_text2tag(data_path):
    """
    cluster based on the text to tag
    """
    f = open(data_path)
    ttrs_array = json.load(f)
    f.close()
    clusters = []
    # find clusters using threshold technique
    for k in ttrs_array.keys():
        if len(ttrs_array[k].values()) == 0: continue
        mean_ttrs = statistics.mean(list(ttrs_array[k].values()))
        std_ttrs = statistics.pstdev(list(ttrs_array[k].values()))
        extracted_text = []
        for key,value in ttrs_array[k].items():
            # choose pairs with over 2 standard deviation
            if (value > (mean_ttrs + 1.0 * std_ttrs)):
                extracted_text.append(key)
        clusters.append(extracted_text)

    return clusters

def combine_ttr_res_with_hw1_output(ttr_res, hw1_out_path, save_path):
    """
    add the ttr'ed result to the hw1_out_path
    """
    hw1_output = pd.read_csv(hw1_out_path, sep='\t', header=0)

    hw1_output["TTR'ed extracted text"] = ttr_res[1:]

    # hw1_output.to_csv(save_path, sep='\t',index=False)
    return hw1_output

if __name__ == "__main__":
    # part3a_res = text2tag(os.getcwd() + "/data/emails_context.json")
    # print(part3a_res[2582])
    
    
    part3_ttr_res = cluster_based_text2tag(os.getcwd() + "/data/emails_ttr_scores.json")
    
    hw1_output_path = os.getcwd() + "/data/final_data_v2.tsv"
    part3_save_path = os.getcwd() + "/data/hw2_part3.tsv"
    combine_ttr_res_with_hw1_output(part3_ttr_res, hw1_output_path, part3_save_path)

   