import os
import sys
import pandas as pd

dir = sys.argv[1] if len(sys.argv) > 1 else './All_Raw_Data'
out = sys.argv[2] if len(sys.argv) > 2 else 'data/pruned_data.csv'

cols = ['Date', 'Page Type', 'Author', 'Full Text', 'Gender', 'Hashtags', 'Thread Entry Type', 'Twitter Followers', 'Twitter Following', 'Twitter Tweets', 'Twitter Verified', 'Account Type', 'Impact', 'Impressions', 'Professions', 'Reach (new)', 'Region']

data = {col: [] for col in cols}
df_full = pd.DataFrame(data)
df_full.to_csv(out, sep='\t', mode='a', index=False)

start = 0
count = start
for filename in os.listdir(dir)[start:]:
    f = os.path.join(dir, filename)
    print('file ', count)
    if os.path.isfile(f) and filename[-4:] == 'xlsx':
        print(filename)
        df = pd.read_excel(f, header=6, usecols=cols)
        df_new = df[df['Page Type'] == 'twitter']
        print('writing ', filename)
        df_new.to_csv(out, sep='\t', header=False, mode='a', index=False)
        print('wrote ', filename)
    count += 1
