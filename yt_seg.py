from config import get_config
from raw_dataset import get_raw_dataset

from pytube import YouTube
import cv2
import sys
import subprocess as sp
import numpy as np
import scipy.signal as signal
from os import remove as rm
from os.path import exists
from matplotlib import pyplot as plt
from scipy.io import wavfile

def cleanString(x):
	chars = '\\/:*?"<>|'
	for c in chars:
		x = x.replace(c, '')
	return x

class YT_SEG_Converter():
	def __init__(self,
				  DS,
				  SAVE_DIR,
				  CLIP_LENGTH=10,
				  SEGMENT_LENGTH=1,
				  NUM_SEGMENTS=9):
		self.SAVE_DIR = SAVE_DIR
		self.CLIP_LENGTH = CLIP_LENGTH
		self.SEGMENT_LENGTH = SEGMENT_LENGTH
		self.NUM_SEGMENTS = NUM_SEGMENTS
		self.config = get_config('config.ini')
		self.DS = DS
		ds = get_raw_dataset(self.config['raw_dataset_FN'])
		self.ST_by_id = dict()
		print('reading raw dataset start times')
		for i in range(len(ds)):
			self.ST_by_id[ds['id'][i]] = ds['start'][i]


	def download_audio(self, id, idx, i):
	
		fo = '{}/{}_audio_{}.mp4'.format(self.SAVE_DIR, idx, cleanString(id))

		if not exists(fo):
			try:
				yt = YouTube('https://www.youtube.com/watch?v=' + id)
				audio_streams = list(yt.streams.filter(type='audio', file_extension='mp4'))
				audio_streams.sort(key=lambda x: int(x.abr[:-4]))
				audio_fp = audio_streams[-1].download(filename=fo)
			except Exception as e:
				audio_fp = e
		else:
			audio_fp = fo

		return audio_fp

	def clip_audio(self, audio_fp, start_time, id, idx, i):
		fo = '{}/{}-{}_{}_audio_{}.mp3'.format(self.SAVE_DIR,idx, i, start_time, cleanString(id))
		if not exists(fo):
			sp.run(['ffmpeg', '-i', audio_fp, '-ss', str(start_time), '-t', '4', fo], \
				capture_output=True, timeout=5)

	def get_segment(self, idx, i):
		id = self.DS.id(idx)
		print('getting audio {} {}...'.format(idx, id), end='')
		audio_fp = self.download_audio(id, idx, i)
		self.clip_audio(audio_fp, max(self.ST_by_id[id]-1.5+i, 0), id, idx, i)
		print('done')

		
	