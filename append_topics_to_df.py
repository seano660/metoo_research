import pandas as pd
import sys

data_path, topic_path, num_topics = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = pd.read_csv(data_path, sep='\t')
topics = pd.read_csv(topic_path, header=None, sep='\t')

for i in range(num_topics):
    data[str(i)] = topics[i]

data.to_csv(f'data_with_{num_topics}.csv', sep='\t', index=False)
