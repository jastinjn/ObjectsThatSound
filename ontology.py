import json
import pandas as pd
import pickle
from os.path import exists

#FILENAME = "C:/Users/Joel Huang/Projects/EECS 442 Project/ontology.json"
#ANA_FN = "C:/Users/Joel Huang/Projects/EECS 442 Project/items_per_class.csv"
def get_ontology(FILENAME):
	pickleFP = FILENAME[:-5] + '.pickle'
	if exists(pickleFP):
		f = open(pickleFP, 'rb')
		classes_by_id, classes_by_name = pickle.load(f)
		f.close()
	else:
		f = open(FILENAME, 'r')
		classes = json.load(f)
		f.close()

		classes_by_id = dict()
		classes_by_name = dict()

		for c in classes:
			classes_by_id[c['id']] = c['name']
			classes_by_name[c['name']] = c['id']
			#print(classes_by_name[c['name']], classes_by_id[c['id']])
		f = open(pickleFP, 'wb')
		pickle.dump((classes_by_id, classes_by_name), f)
		f.close()

	return classes_by_id, classes_by_name