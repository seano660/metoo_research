import gensim
import gensim.corpora as corpora

from pprint import pprint

data_path = 'data/training_data.txt'
num_topics = [25]#, 50, 75, 100, 125, 150, 175, 200]
decays = [0.5]#, 0.6, 0.7, 0.8, 0.9]
with open(data_path, 'r', encoding='utf-8') as f:
    data = f.readlines()

print('loaded data')

data = [text.strip().split() for text in data]

id2word = corpora.Dictionary(data)

texts = data

corpus = [id2word.doc2bow(text) for text in texts]

print('created corpus')
for num in num_topics:
    for decay in decays:
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num,
                                                    decay=decay,
                                                    random_state=100)

        print(f'trained model topics-{num}, decay-{decay}')
        pprint(lda_model.print_topics())

        lda_model.save('models/lda-model-' + str(num) + '-' + str(decay))
        print('saved model\n')
        with open('training_results.txt', 'a') as f:
            f.write(f'topics={num}, decay={decay}\n')
            f.write(str(lda_model.print_topics()))
            f.write('\n')

