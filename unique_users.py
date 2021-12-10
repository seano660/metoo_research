import pandas as pd

df = pd.read_csv('data/pruned_data.csv', sep='\t')
users = set(df['Author'].values.tolist())

with open('data/unique_users.txt', 'w') as f:
    for user in users:
        f.write(user + '\n')
