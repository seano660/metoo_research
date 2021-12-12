import pandas as pd
from prepare_training import preprocess_text

df = pd.read_csv('data/data_with_demographic.csv', sep='\t')
df = df[df['UserType'].notna() | df['Account Type'].notna()]
print(df)

df.to_csv('data/analysis_data.csv', sep='\t', index=False)

data = df['Full Text'].values.tolist()
df = None
data = preprocess_text(data)

data = [" ".join(text + ['\n']) for text in data]

with open('data/analysis_text.txt', 'w', encoding='utf-8') as f:
    f.writelines(data)

