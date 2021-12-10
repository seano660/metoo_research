import json
import pandas as pd
import sys

def append_demographic(json_name, data_name, name):
    with open(json_name) as f:
        j = json.load(f)
    df = pd.read_csv(data_name, sep='\t')
    df[name] = df['Author'].map(lambda x: j.get(x))
    df.to_csv('data/data_with_demographic.csv', sep='\t', index=False)


if __name__ == '__main__':
    j = sys.argv[1]
    data = sys.argv[2]
    name = sys.argv[3]
    append_demographic(j, data, name)
    print('appended', name)

