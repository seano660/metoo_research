python3 prune.py 
python3 unique_users.py
python3 identify_genders.py
python3 append_demographic.py data/user2gender.json data/pruned_data.csv gender_inferred
python3 append_demographic.py data/user2type.json data/data_with_demographic.csv UserType
python3 append_demographic.py data/user2ethnicity.json data/data_with_demographic.csv Ethnicity
python3 prepare_training.py data/data_with_demographic.csv data/training_data.txt
python3 train_lda.py
python3 get_analysis_set.py
python3 text2corpus.py
python3 identify_topics.py models/lda-model-100-0.7 100
python3 append_topics_to_df.py data/analysis_data.csv data/100_sentence_topics.csv 100 
python3 topics2json.py models/lda-model-100-0.7 100



