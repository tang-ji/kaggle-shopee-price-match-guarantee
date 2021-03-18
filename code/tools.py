import os
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import unidecode
import codecs

# import spacy
# from wordcloud import WordCloud, STOPWORDS
# from fuzzywuzzy import fuzz
# import cudf
# import cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.common.sparsefuncs import csr_row_normalize_l2

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

import gc

def plot_bar_chart(x, y, title, rotation_angle=45):
    plt.figure(figsize = (20, 15))
    sns.barplot(x=x, y=y).set_title(title)
    plt.xticks(rotation=rotation_angle)
    plt.show()
    

def plot_images(dataframe, column_name, value):
    '''
    Plot images using image_path, based on the column & value filter
    '''
    plt.figure(figsize = (30, 30))
    value_filter = dataframe[dataframe[column_name] == value]
    image_paths = value_filter['image_path'].to_list()
    print(f'Total images: {len(image_paths)}')
    posting_id = dataframe['posting_id'].to_list()
    for i, j in enumerate(zip(image_paths, posting_id)):
        plt.subplot(10, 10, i + 1)
        img = cv2.cvtColor(cv2.imread(j[0]), cv2.COLOR_BGR2RGB)
        plt.title(j[1])
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(img)

        
def plot_matched_images(images_path, posting_id):
    plt.figure(figsize = (50, 50))
    for i, j in enumerate(zip(images_path, posting_id)):
        plt.subplot(10, 10, i + 1)
        img = cv2.cvtColor(cv2.imread(j[0]), cv2.COLOR_BGR2RGB)
        plt.title(j[1])
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(img)
        
        
def plot_images_by_label_group(label):
    plt.figure(figsize = (30, 30))
    label_filter = train_df[train_df['label_group'] == label]
    image_paths = label_filter['image_path'].to_list()
    print(f'Total images: {len(image_paths)}')
    posting_id = label_filter['posting_id'].to_list()
    for i, j in enumerate(zip(image_paths, posting_id)):
        plt.subplot(10, 10, i + 1)
        img = cv2.cvtColor(cv2.imread(j[0]), cv2.COLOR_BGR2RGB)
        plt.title(j[1])
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(img)
        
        
def plot_images_by_phash(image_phash):
    '''
    Plots image by phash value from train_df dataframe
    '''
    plt.figure(figsize = (30, 30))
    phash_filter = train_df[train_df['image_phash'] == image_phash]
    image_paths = phash_filter['image_path'].to_list()
    print(f'Total images: {len(image_paths)}')
    posting_id = phash_filter['posting_id'].to_list()
    for i, j in enumerate(zip(image_paths, posting_id)):
        plt.subplot(10, 10, i + 1)
        img = cv2.cvtColor(cv2.imread(j[0]), cv2.COLOR_BGR2RGB)
        plt.title(j[1])
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(img)
        
        
def hamming_distance(phash1, phash2):
    '''
    helper function to calculate phash similarity
    '''
    phash1 = bin(int(phash1, 16))[2:].zfill(64)
    phash2 = bin(int(phash2, 16))[2:].zfill(64)
    distance = np.sum([i != j for i, j in zip(phash1, phash2)])
    return distance


def hamming_distance_bin(phash1, phash2):
    '''
    helper function to calculate phash similarity
    '''
    return np.sum([i != j for i, j in zip(phash1, phash2)])


def get_record_from_df(dataframe, column_name, value):
    '''
    Returns records from dataframe for the given value & column
    '''
    return dataframe[dataframe[column_name] == value]
    
    
def cosine_similarity(string1, string2):
    d1 = nlp(string1)
    d2 = nlp(string2)
    return d2.similarity(d2)


def find_matches(posting_id, dataframe, dist_thr=10, title_thr=60):
    '''
    posting_id: posting_id 
    dataframe: train/test dataframe from which the phash & title can be pulled
    dist_thr: phash distance/score threshold
    title_thr: title score threshold from 100
    '''
    results = {}
    phash_value = dataframe[dataframe['posting_id'] == posting_id].image_phash.to_list()[0]
    title_value = dataframe[dataframe['posting_id'] == posting_id].clean_title.to_list()[0]
    print(title_value)
    for i in dataframe.itertuples():
        phash_dist = hamming_distance(phash_value, i.image_phash)
        title_score = fuzz.token_set_ratio(title_value.lower(), i.clean_title.lower())

        if phash_dist <= dist_thr:
            # print(i.posting_id, " ::: ", i.title, phash_dist)
            # results.append([i.posting_id, i.image_path])
            results[i.posting_id] = i.image_path
            continue
        
        if title_score > title_thr:
            # print(i.posting_id, " ::: ", i.title, title_score)
            # results.append([i.posting_id, i.image_path])
            results[i.posting_id] = i.image_path
    return results


class ProductMatch:
    '''
    Aggregating phash | fuzzymatch | cosine similarity
    '''
    def __init__(self, cudf_df, pro_df):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(cudf_df['clean_title'])
        self.pro_df = pro_df
        
        
    def find_phash_fuzz_match(self, posting_id, dist_thr=10, title_thr=60):
        phash_val = self.pro_df.loc[self.pro_df['posting_id'] == posting_id].hash.to_list()[0]
        title_val = self.pro_df.loc[self.pro_df['posting_id'] == posting_id].clean_title.to_list()[0]

        self.pro_df['image_phash_score'] = self.pro_df['hash'].apply(lambda x: hamming_distance_bin(phash_val, x))
        self.pro_df['title_score'] = self.pro_df['clean_title'].apply(lambda x: fuzz.token_set_ratio(title_val, x))
        self.pro_df.sort_values(by='title_score', ascending=False, inplace=True)
        i_score = self.pro_df.loc[self.pro_df['image_phash_score'] <= dist_thr]
        t_score = self.pro_df.loc[self.pro_df['title_score'] > title_thr]

        self.fuz_ph = {**dict(zip(i_score.posting_id.to_list()[:50], i_score.image_path.to_list()[:50])), **dict(zip(
            t_score.posting_id.to_list()[:50], t_score.image_path.to_list()[:50]))}

        return self.fuz_ph
    
    
    # Ref: https://medium.com/rapids-ai/natural-language-processing-text-preprocessing-and-vectorizing-at-rocking-speed-with-rapids-cuml-74b8d751812e
    def efficient_csr_cosine_similarity(self, query, matrix_normalized=False):
        query = csr_row_normalize_l2(query, inplace=False)
        if not matrix_normalized:
            self.tfidf_matrix = csr_row_normalize_l2(self.tfidf_matrix, inplace=False)
        return self.tfidf_matrix.dot(query.T)

    def cos_match(self, df, query, cos_thr=0.2, top_n=50):
        query = self.pro_df.loc[self.pro_df['posting_id'] == query].clean_title.to_list()[0]
        query_vec = self.vectorizer.transform(cudf.Series([query]))
        similarities = self.efficient_csr_cosine_similarity(query_vec, matrix_normalized=True)
        similarities = similarities.todense().reshape(-1)
        best_idx = similarities.argsort()[-top_n:][::-1]
        op_df = cudf.DataFrame({
            'posting_id': df['posting_id'].iloc[best_idx],
            # 'title': df['clean_title'].iloc[best_idx],
            'image_path': df['image_path'].iloc[best_idx],
            'similarity': similarities[best_idx]
        })
        cos_df = op_df.to_pandas()
        cos_df = cos_df[~cos_df['posting_id'].isin([list(self.fuz_ph.keys())])]
        cos_df = cos_df.loc[cos_df['similarity'] > cos_thr]
        cos_df = dict(zip(cos_df.posting_id.to_list()[:50 - len(self.fuz_ph.keys())], cos_df.image_path.to_list()[:50 - len(self.fuz_ph.keys())]))
        return cos_df
    
    
# Ref: https://medium.com/rapids-ai/natural-language-processing-text-preprocessing-and-vectorizing-at-rocking-speed-with-rapids-cuml-74b8d751812e

def efficient_csr_cosine_similarity(query, tfidf_matrix, matrix_normalized=False):
    query = csr_row_normalize_l2(query, inplace=False)
    if not matrix_normalized:
        tfidf_matrix = csr_row_normalize_l2(tfidf_matrix, inplace=False)
    return tfidf_matrix.dot(query.T)

def product_match(df, query, vectorizer, tfidf_matrix, top_n=50):
    print(f"Product match: {query}")
    query_vec = vectorizer.transform(cudf.Series([query]))
    similarities = efficient_csr_cosine_similarity(query_vec, tfidf_matrix, matrix_normalized=True)
    similarities = similarities.todense().reshape(-1)
    best_idx = similarities.argsort()[-top_n:][::-1]
    op_df = cudf.DataFrame({
        'posting_id': df['posting_id'].iloc[best_idx],
        'title': df['clean_title'].iloc[best_idx],
        'image_path': df['image_path'].iloc[best_idx],
        'similarity': similarities[best_idx]
    })
    return op_df


digit_check = re.compile('\d')
def check_alpha_num(token):
    # check if the token id alphanumeric
    return bool(digit_check.search(token))


def handle_consecutive_char(string):
    # check & fix for 3 or more consecutive characters
    return re.sub(r'(.)\1+\1+', r'\1', string)