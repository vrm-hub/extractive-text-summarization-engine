import os
import utility as ut

directory = '../processed_updated'
COUNT = 50
limit = 0
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if limit > COUNT:
        break
    ut.read_pickle_data(file_path)

    limit += 1
