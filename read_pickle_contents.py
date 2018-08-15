import pickle
import pprint

with open('lineClasses.txt', 'rb') as f:
    line_classes_data = pickle.load(f)
    pprint.pprint(line_classes_data)
