import json
import gensim
import sys

ldamodel, num_topics = gensim.models.ldamodel.LdaModel.load(sys.argv[1]), int(sys.argv[2])
prec_words = {i: [x[0] for x in ldamodel.show_topic(i, 10)] for i in range(num_topics)}

with open(f'{num_topics}-topic-words.json', 'w') as f:
    json.dump(prec_words, f)