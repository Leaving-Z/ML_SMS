# data_processing.py
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# 1) 数据加载
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])
    return df

# 2) 数据清洗
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(t):
    t = re.sub(r'[^a-z\s]', ' ', t.lower())
    return ' '.join(stemmer.stem(w) for w in word_tokenize(t) if w not in stop_words)

def preprocess_data(df):
    df['clean'] = df['text'].apply(clean_text)
    return df

# 3) 特征提取
def extract_features(df):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X = tfidf.fit_transform(df['clean'])
    y = (df['label'] == 'spam').astype(int)
    return X, y
