#FILENAME = "C:/Users/Joel Huang/Projects/EECS 442 Project/classes.txt"

def get_use_classes(FILENAME):
	f = open(FILENAME, 'r')
	line = f.readline()
	f.close()
	
	use_classes = set()
	for c in line.split("; "):
		use_classes.add(c)
	return use_classes