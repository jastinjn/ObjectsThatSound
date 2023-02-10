from dataset import download_dataset
#import numpy as np
#SAVE_DIR = "C:/Users/Joel Huang/Projects/EECS 442 Project/"

#x = np.load('{}/data/spectrograms/--1XHaNcX2Y.npz'.format(SAVE_DIR))
#for k,v in x.items():
#	print(k,v)

#from yt_npz import YT_NPZ_Converter

#YN = YT_NPZ_Converter(SAVE_DIR)
#YN.convert('--34LejG4cE', 20)

SAVE_DIR = "C:/Users/Joel Huang/Projects/EECS 442 Project/"
raw_dataset_FN		= "C:/Users/Joel Huang/Projects/EECS 442 Project/unbalanced_train_segments.csv"
ontology_FN			= "C:/Users/Joel Huang/Projects/EECS 442 Project/ontology.json"
use_classes_FN		= "C:/Users/Joel Huang/Projects/EECS 442 Project/classes.txt"
download_dataset(SAVE_DIR, raw_dataset_FN, ontology_FN, use_classes_FN)
