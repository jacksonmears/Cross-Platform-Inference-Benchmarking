

import os


path = "ground_truths"

files = [file for file in os.listdir(path) if file[-1]=='z']

print(files[-1])
