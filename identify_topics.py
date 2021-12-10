import pandas as pd
import gensim
import sys
import math
import json

def identify_topics(corpus, ldamodel, num_topics):
    times = math.ceil(len(corpus) / 1000)
    for _ in range(times):
        print(_ / times, ' done')
        sent_topics_df = pd.DataFrame({i: [] for i in range(num_topics)})
        start, end = _*1000, (1+_)*1000
        for i, row in enumerate(ldamodel[corpus[start:end]]):
            temp = [0] * num_topics
            for topic in row:
                temp[topic[0]] = topic[1]
            sent_topics_df = sent_topics_df.append(pd.Series(temp), ignore_index=True)
        sent_topics_df.to_csv(f'{num_topics}_sentence_topics.csv', sep='\t', index=False, mode='a', header=False)
        print(f'saved {start}-{end}')

if __name__ == '__main__':
    print('loaded prereqs')
    model_path, num_topics = sys.argv[1], int(sys.argv[2])
    with open('data/corpus.json') as f:
        data = json.load(f)
    print('loaded data')
    ldamodel = gensim.models.ldamodel.LdaModel.load(model_path)
    print('loaded model')
    identify_topics(data, ldamodel, num_topics)