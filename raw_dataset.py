import pandas as pd
import numpy as np

#FILENAME = "C:/Users/Joel Huang/Projects/EECS 442 Project/unbalanced_train_segments.csv"
sep = ','
def get_raw_dataset(FILENAME):
	f = open(FILENAME, "r")
	rawlist = f.readlines()
	f.close()
	rawlist = rawlist[3:]

	n = len(rawlist)
	ids = [None]*n
	start = np.zeros(n, dtype=int)
	end = np.zeros(n, dtype=int)
	classes = [None]*n

	for i,line in enumerate(rawlist):
		_1 = line.find(sep)
		_2 = line.find(sep, _1+1)
		_3 = line.find(sep, _2+1)
		ids[i] = line[:_1]
		start[i] = int(float(line[_1+2:_2]))
		end[i] = int(float(line[_2+2:_3]))
		cls_set = set()
		for s in line[_3+3:-2].split(','):
			cls_set.add(s)
		classes[i] = cls_set
	return pd.DataFrame({'id':ids, 'start':start, 'end':end, 'classes':classes})
	