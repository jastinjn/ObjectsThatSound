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

class YT_NPZ_Converter():
	def __init__(self,
			  SAVE_DIR,
			  FRAME_HEIGHT=128,
			  SPEC_SIZE=(128,128),
			  CLIP_LENGTH=10,
			  SEGMENT_LENGTH=1,
			  NUM_SEGMENTS=9,
			  SAMPLING_RATE=44100):
		self.SAVE_DIR = SAVE_DIR
		self.FRAME_HEIGHT = FRAME_HEIGHT
		self.SPEC_SIZE = SPEC_SIZE
		self.CLIP_LENGTH = CLIP_LENGTH
		self.SEGMENT_LENGTH = SEGMENT_LENGTH
		self.NUM_SEGMENTS = NUM_SEGMENTS
		self.SAMPLING_RATE = SAMPLING_RATE
		
	def convert(self, id, start_time):
		video_fp, audio_fp = self.download_video(id)

		if type(video_fp) != str or type(audio_fp) != str:
			if type(video_fp) == str:
				rm(video_fp)
			if type(audio_fp) == str:
				rm(audio_fp)
			return video_fp, audio_fp
		flag, info = self.extract_segments(video_fp, audio_fp, start_time, id)

		audio_fp_new = audio_fp[:-3] + 'wav'
		if exists(video_fp):
			rm(video_fp)
		if exists(audio_fp):
			rm(audio_fp)
		if exists(audio_fp_new):
			rm(audio_fp_new)

		if not flag:
			return info, None

		return True

	def download_video(self, id):
		yt = YouTube('https://www.youtube.com/watch?v=' + id)
		try:
			video_streams = list(yt.streams.filter(type='video', file_extension='mp4'))
			audio_streams = list(yt.streams.filter(type='audio', file_extension='mp4'))

			video_streams.sort(key=lambda x: int(x.resolution[:-1]))	# sort by resolution
			audio_streams.sort(key=lambda x: int(x.abr[:-4]))			# sort by bitrate
		except Exception as e:
			return e, None
	
		fo = cleanString(yt.title) + '.mp4'
		
		if not exists('v_' + fo):
			for i,v in enumerate(video_streams):
				if int(v.resolution[:-1]) >= self.FRAME_HEIGHT or i == len(video_streams)-1:
					# get lowest res that is larger than FRAME_HEIGHT
					try:
						video_fp = v.download(filename='v_' + fo)
					except Exception as e:
						video_fp = e
					break
		else:
			video_fp = 'v_' + fo

		if not exists('a_' + fo):
			try:
				audio_fp = audio_streams[0].download(filename='a_' + fo)	# get lowest bitrate - 44.1KHz
			except Exception as e:
				audio_fp = e
		else:
			audio_fp = 'a_' + fo

		return video_fp, audio_fp


	def extract_segments(self, video_fp, audio_fp, start_time, id):
		audio_fp_new =  audio_fp[:-3] + 'wav'

		try:
			cap = cv2.VideoCapture(video_fp)
			if not cap.isOpened():
				return False, 'cv2 cannot capture {}. Video seems to be corrupted'.format(id)
			total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			fps = cap.get(cv2.CAP_PROP_FPS)
			if fps == 0 or (start_time + self.NUM_SEGMENTS + 1) > total_frame_count / fps:
				return False, 'cv2 detects 0 fps or insufficient frames for {}'.format(id)
		except cv2.error as e:
			return False, e

		if not exists(audio_fp_new):
			try:
				sp.run(['ffmpeg', '-i', audio_fp, '-f', 'wav', audio_fp_new], capture_output=True, timeout=5)
				# If this times out, something is going wrong with the conversion.
				# Change capture_ouput to false to get some debug on stdout
			except Exception as e:
				cap.release()
				return False, e
		try:
			sr, aud = wavfile.read(audio_fp_new)
		except Exception as e:
			cap.release()
			return False, e
		if sr != self.SAMPLING_RATE:
			# We need the sampling rate to be identical for all audio so that the spectrogram makes sense
			return False, "can't use {}. sampling rate is {}".format(audio_fp_new, sr)

		specs = {}
		frames = {}
		for s in range(self.NUM_SEGMENTS):
			cap.set(cv2.CAP_PROP_POS_MSEC, (s+start_time + 0.5*self.SEGMENT_LENGTH) * 1000)
			success, frame = cap.read()
			if not success or frame is None:
				cap.release()
				return False, 'Unable to extract frame {} from {} in {}'.format(s, video_fp, id)

			try:
				frame = cv2.resize(frame, (self.FRAME_HEIGHT, self.FRAME_HEIGHT))	# Resize to correct size for CNN
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)						# Convert from BGR (cv2 encoding) to RGB
				frame = np.transpose(frame, axes=(2,0,1))							# transpose axis from numpy/cv2 style to torch channel style
			except cv2.error as e:
				cap.release()
				return False, e

			start_idx = (s+start_time) * self.SAMPLING_RATE
			end_idx = start_idx + self.SEGMENT_LENGTH * self.SAMPLING_RATE

			aud_seg = np.sum(aud[start_idx:end_idx], axis=1)/2
			#wavfile.write('{}.wav'.format(s), SAMPLING_RATE, np.int16(aud_seg))

			_,_,spec = signal.spectrogram(
				aud_seg,
				self.SAMPLING_RATE,
				nperseg=self.SAMPLING_RATE//100,			# 0.01 s
				noverlap=(self.SAMPLING_RATE/100)//4)		# quarter overlap

			if spec.shape != self.SPEC_SIZE:
				try:
					spec = cv2.resize(spec, self.SPEC_SIZE)
				except cv2.error as e:
					cap.release()
					return False, e
			if spec.shape != self.SPEC_SIZE:
				cap.release()
				return False, 'Unable to extract spectrogram {} from {} in {}'.format(s, audio_fp, id)
			
			frames[str(s)] = frame
			specs[str(s)] = spec

		cap.release()
		np.savez_compressed('{}/data/frames/{}'.format(self.SAVE_DIR,id), **frames)
		np.savez_compressed('{}/data/spectrograms/{}'.format(self.SAVE_DIR,id), **specs)

		return True, None


		
	