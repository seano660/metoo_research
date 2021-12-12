from gender_neural import NeuralGenderDemographer
import json


model = NeuralGenderDemographer()

user2gender = {}

genders = {'man': 'male', 'woman': 'female'}

with open('data/unique_users.txt') as f:
    users = f.readlines()
users = [u.strip() for u in users]
    
for user in users:
    gender = genders[model.process_tweet({'name': user})['gender_neural']['value']]
    user2gender[user] = gender

with open('data/user2gender.json', 'w') as f:
    json.dump(user2gender, f)


