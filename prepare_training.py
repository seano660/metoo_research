import sys
import pandas as pd
import re
from gensim.utils import simple_preprocess

from nltk.corpus import stopwords

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['metoo', 'rt'])
    return [[w for w in text if w not in stop_words] for text in texts]

def remove_mentions(data):
    return [re.sub('@\S*\s?', '', str(sent)) for sent in data]

def remove_links(data):
    return [re.sub('http\S*\s?', '', str(sent)) for sent in data]

def remove_newlines(data):
    return [re.sub('\s+', ' ', str(sent)) for sent in data]

def remove_singlequotes(data):
    return [re.sub("\'", "", str(sent)) for sent in data]

def preprocess_text(data):
    # remove mentions and links
    data = remove_mentions(data)
    data = remove_links(data)

    data = remove_newlines(data)
    data = remove_singlequotes(data)
    print('completed regex\n')
    data = [simple_preprocess(tweet, deacc=True) for tweet in data]
    print('completed simple preprocess\n')
    data = remove_stopwords(data)
    return data

def load_data(data_path):
    df = pd.read_csv(data_path, sep='\t', usecols=['Full Text', 'Thread Entry Type'])
    return df[df['Thread Entry Type'] != 'share']['Full Text'].values.tolist()

if __name__ == '__main__':
    data_path, output_name = sys.argv[1], sys.argv[2]

    data = load_data(data_path)
    print('loaded data\n')
    
    data = preprocess_text(data)
    print('completed preprocessing\n')
    data = [" ".join(text + ['\n']) for text in data]

    with open(output_name, 'w', encoding='utf-8') as f:
        f.writelines(data)

