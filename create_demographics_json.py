import json
import pandas as pd
import sys

data_path = sys.argv[1]

data = pd.read_csv(data_path, sep='\t')

grouped = data.groupby('Author')
authors = [author[0] for author in grouped['Author']]
genders = [gender[1].tolist()[0] for gender in grouped['gender_inferred']]
ethnicities = [eth[1].tolist()[0] for eth in grouped['Ethnicity']]
account_types = [user_type[1].tolist()[0] if user_type[1].tolist()[0] else acc_type[1].tolist()[0] for user_type, acc_type in zip(grouped['UserType'], grouped['Account Type'])]
intersectional = [f'{eth} {gender}' if eth and gender else None for eth, gender in zip(ethnicities, genders)]

output = {author: {'gender': gender, 'ethnicity': ethnicity, 'Type': acc_type, 'Intersectional': intersect} for author, gender, ethnicity, acc_type, intersect in zip(authors, genders, ethnicities, account_types, intersectional)}

with open('user2demographics.py', 'w') as f:
    json.dump(output, f)




