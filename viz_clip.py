from dataset import AudioSet
from config import get_config
import visualize as v
import sys

import matplotlib.pyplot as plt

import numpy as np


from torch.utils.data import DataLoader
cfg = get_config('config.ini')

size=None
tAS = AudioSet('test', cfg['SAVE_DIR'], VAL_RATIO=cfg['VAL_RATIO'], TEST_RATIO=cfg['TEST_RATIO'], FT_RATIO=cfg['FT_RATIO'], size=size)

args = sys.argv[1:]

while True:
	v.viz_clip(tAS, int(args[0]))
	if len(args) > 1:
		v.viz_clip(tAS, int(args[1]))
	plt.show()
	args = input('-').split()