import gensim
import json
with open('data/analysis_text.txt', encoding='utf-8') as f:
    data = f.readlines()

data = [sent.strip().split() for sent in data]

ldamodel = gensim.models.ldamodel.LdaModel.load('models/lda-model-100-0.7')
id2word = ldamodel.id2word

corpus = [id2word.doc2bow(text) for text in data]
with open('data/corpus.json', 'w') as f:
    json.dump(corpus, f)



