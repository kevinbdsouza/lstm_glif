import os
from os import listdir
from os.path import isfile, join

data_path = ""

test_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
test_files.sort()

# os.rename(old_file, new_file)
