import json

def get_config(FILENAME):
	f = open(FILENAME, 'r')
	config = json.load(f)
	f.close()

	return config