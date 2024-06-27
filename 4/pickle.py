import pickle

with open('ER_data_bet.pickle', 'rb') as f:
    data_list = pickle.load(f)

for item in data_list:
    print(item)
