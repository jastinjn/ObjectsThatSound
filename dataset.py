from yt_npz import YT_NPZ_Converter
from ontology import get_ontology
from raw_dataset import get_raw_dataset
from use_classes import get_use_classes

import glob
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from os.path import exists
from os import mkdir
from os import sep
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

FRAME_HEIGHT = 128
SPEC_SIZE = (128, 128)
CLIP_LENGTH = 10
SEGMENT_LENGTH = 1
NUM_SEGMENTS = 9
SAMPLING_RATE = 44100
SPEC_MEAN_STD_FP = "SPEC_MEAN_STD.pickle"


def download_dataset(SAVE_DIR, raw_dataset_FN, ontology_FN, use_classes_FN, re_attempt_failed_dl=False):
	YN = YT_NPZ_Converter(SAVE_DIR, FRAME_HEIGHT, SPEC_SIZE, CLIP_LENGTH, SEGMENT_LENGTH, NUM_SEGMENTS, SAMPLING_RATE)

	print('Getting dataset info...')
	raw_ds = get_raw_dataset(raw_dataset_FN)
	classes_by_id, classes_by_name = get_ontology(ontology_FN)
	use_classes = get_use_classes(use_classes_FN)

	mask = np.zeros(len(raw_ds), dtype=bool)
	for i in range(len(mask)):
		for cls_id in raw_ds['classes'][i]:
			if classes_by_id[cls_id] in use_classes:
				mask[i] = True
				break
	
	ds = raw_ds[mask]
	print('Dataset Size: {}'.format(len(ds)))

	if not exists('{}/data'.format(SAVE_DIR)):
		mkdir('{}/data'.format(SAVE_DIR))
	if not exists('{}/data/frames'.format(SAVE_DIR)):
		mkdir('{}/data/frames'.format(SAVE_DIR))
	if not exists('{}/data/spectrograms'.format(SAVE_DIR)):
		mkdir('{}/data/spectrograms'.format(SAVE_DIR))

	prior_vids = glob.glob('{}/data/frames/*.npz'.format(SAVE_DIR))
	prior_auds = glob.glob('{}/data/spectrograms/*.npz'.format(SAVE_DIR))
	failed_vids = glob.glob('{}/data/frames/*.failed'.format(SAVE_DIR))
	failed_auds = glob.glob('{}/data/spectrograms/*.failed'.format(SAVE_DIR))
	
	pv_set = set()
	pa_set = set()
	fv_set = set()
	fa_set = set()
	for i in prior_vids:
		pv_set.add(i[i.rfind(sep)+1:-4])
	for i in prior_auds:
		pa_set.add(i[i.rfind(sep)+1:-4])
	for i in failed_vids:
		fv_set.add(i[i.rfind(sep)+1:-7])
	for i in failed_auds:
		fa_set.add(i[i.rfind(sep)+1:-7])
	
	prior_set = set.intersection(pv_set, pa_set)
	failed_set = set.intersection(fv_set, fa_set)

	failed = 0
	prior_failed = 0
	prior = 0
	successful = 0
	exceptions = {}
	print('create a file called "stop" in content directory to halt download safely')
	idx = ds.index
	for i in tqdm(idx, desc='Downloading dataset'):
		id = ds['id'][i]
		if id in prior_set:
			prior += 1
		elif not re_attempt_failed_dl and id in failed_set:
			prior_failed += 1
		else:
			res = YN.convert(id, ds['start'][i])
			if type(res) == bool:
				successful += 1
			else:
				failed += 1
				exceptions[id] = res
				f = open('{}/data/frames/{}.failed'.format(SAVE_DIR, id), 'w')
				f.close()
				f = open('{}/data/spectrograms/{}.failed'.format(SAVE_DIR, id), 'w')
				f.close()

		if exists('stop'):
			print('\nHalting download')
			break
	print('{} Successful downloads'.format(successful))
	print('{} Prior downloads skipped'.format(prior))
	print('{} Prior failed downloads skipped'.format(prior_failed))
	print('{} Failed downloads'.format(failed))
	for k,v in exceptions.items():
		if type(v) == tuple:
			print('{}\t{}: {}\t{}: {}'.format(k, type(v[0]), v[0], type(v[1]), v[1]))
		else:
			print('{}\t{}: {}'.format(k, type(v), v))
	print('Successfully exited download. Garbage collector should be running for awhile...')


def identity(x):
	return x

class AudioSet(Dataset):
	'''
	use [] operator to get frame-spec pair with appropriate transforms and labels
	use 


	Init Args:
		mode		train, val, test. Currently does nothing
		SAVE_DIR
		VAL_RATIO	the split eval-train. last VAL_RATIO * len items of dataset will be for eval
		FT_RATIO	must be an int for the whole dataset to be used evenly
					It is the ratio of false pairs to true pairs
		size		Consider only first <size> clips found
	'''
	def __init__(self, mode, SAVE_DIR, VAL_RATIO=0.08, TEST_RATIO=0.02, FT_RATIO=2, size=None):
		self.epsilon = 10**-8

		self.mode = mode
		if mode == 'train':
			self.train = True
		elif mode == 'val' or mode == 'test':
			self.train = False
		else:
			raise ValueError("Argument mode should be either 'train', 'val', or 'test'")

		print('Loading dataset from {}...'.format(SAVE_DIR))
		self.vids_fp = sorted(glob.glob('{}/data/frames/*.npz'.format(SAVE_DIR)))
		self.auds_fp = sorted(glob.glob('{}/data/spectrograms/*.npz'.format(SAVE_DIR)))
		
		assert len(self.vids_fp) == len(self.auds_fp)
		clips_total = len(self.vids_fp)

		if size is not None:
			self.vids_fp = self.vids_fp[:size]
			self.auds_fp = self.auds_fp[:size]

		clips_in_size = len(self.vids_fp)
		
		s_mean, s_std =	self.get_spec_mean_std()

		s1 = int(clips_in_size * (1-VAL_RATIO-TEST_RATIO))
		s2 = int(clips_in_size * (1-TEST_RATIO))
		if self.train:
			self.vids_fp = self.vids_fp[:s1]
			self.auds_fp = self.auds_fp[:s1]
		elif mode == 'val':
			self.vids_fp = self.vids_fp[s1:s2]
			self.auds_fp = self.auds_fp[s1:s2]
		elif mode == 'test':
			self.vids_fp = self.vids_fp[s2:]
			self.auds_fp = self.auds_fp[s2:]

		self.FT_RATIO = FT_RATIO
		self.len = len(self.vids_fp)

		print('{} clips found, using {} of {}'.format(clips_total, self.len, clips_in_size))

		if self.train:
			self.frame_transforms = transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomResizedCrop(FRAME_HEIGHT, scale=(0.8,1)),
				transforms.ColorJitter(brightness=0.1, saturation=0.1),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])
		else:
			self.frame_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

		self.spec_transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[s_mean], std=[s_std])
			])

	def __len__(self):
		return self.len * self.FT_RATIO

	def __getitem__(self, idx, i=None, jdx=None, j=None):
		if idx >= len(self):
			raise IndexError('Index {} is out of bounds for AudioSet of length {} and FT_RATIO {}'\
				.format(idx, self.len, self.FT_RATIO))

		if i is None:
			i = str(np.random.randint(0, high=NUM_SEGMENTS-1))
		if jdx is None:
			# positive sample
			if idx < self.len:
				jdx = idx
				j = i

			# negative sample
			else:
				idx %= self.len
				while True:
					jdx = np.random.randint(0, high=self.len-1)
					if jdx != idx:
						break
				j = str(np.random.randint(0, high=NUM_SEGMENTS-1))
		label = 1 if idx == jdx and i == j else 0

		frame = np.load(self.vids_fp[idx])[i]
		spec = np.load(self.auds_fp[jdx])[j]
		spec[spec == 0] = self.epsilon
		spec = np.log(spec)

		frame = torch.Tensor(frame)
		frame /= 255
		frame = self.frame_transforms(frame)
		spec = self.spec_transforms(spec)
		assert frame.shape == (3, FRAME_HEIGHT, FRAME_HEIGHT)
		assert spec.shape == (1,) + SPEC_SIZE
		return frame, spec, torch.Tensor([label]), torch.Tensor([idx, jdx, int(i), int(j)])


	def view(self, idx, jdx=None):
		if idx >= len(self):
			raise IndexError('Index {} is out of bounds for AudioSet of length {}'.format(idx, len(self)))
		if jdx is not None and jdx >= NUM_SEGMENTS:
			raise IndexError('Index {} is out of bounds for {} segments'.format(jdx, NUM_SEGMENTS))
		idx %= self.len

		frame = dict(np.load(self.vids_fp[idx]))
		spec = dict(np.load(self.auds_fp[idx]))

		if jdx is not None:
			frame = frame[str(jdx)]
			spec = spec[str(jdx)]
			spec[spec == 0] = self.epsilon
			spec = np.log(spec)
		else:
			for key in spec.keys():
				s = spec[key]
				s[s == 0] = self.epsilon
				s = np.log(s)
				spec[key] = s

		return frame, spec

	def id(self, idx):
		if idx >= len(self):
			raise IndexError('Index {} is out of bounds for AudioSet of length {}'.format(idx, len(self)))
		idx %= self.len

		fp = self.vids_fp[idx]
		return fp[fp.rfind(sep)+1:fp.rfind('.')]


	def get_spec_mean_std(self):
		FP = SPEC_MEAN_STD_FP

		length = len(self.auds_fp)
		
		if exists(FP):
			f = open(FP, 'rb')
			auds_fp, mean, std = pickle.load(f)
			f.close()

			if len(auds_fp) == length:
				for i in range(length):
					if auds_fp[i] != self.auds_fp[i]:
						print('Pre-computed spec mean and std were done on a different dataset')
						break
					if i == length - 1 and auds_fp[i] == self.auds_fp[i]:
						# mean and std were precomputed
						return mean, std
			else:
				print('Pre-computed spec mean and std were done on a dataset with length {}, current dataset is length {}'\
					.format(len(auds_fp), length))
		else:
			print('No pre-computed spec mean and std')

		# mean and std not precomputed
		spec_sum = np.zeros(SPEC_SIZE)
				
		for aud_fp in tqdm(self.auds_fp, desc='Computing mean and std of spectrograms'):
			specs = np.load(aud_fp)
			for i in range(NUM_SEGMENTS):
				spec = specs[str(i)]
				spec[spec == 0] = self.epsilon
				spec = np.log(spec)
				spec_sum += spec
		spec_sum /= length * NUM_SEGMENTS

		mean = np.mean(spec_sum)
		std  = np.std(spec_sum)

		# save mean and std
		f = open(FP, 'wb')
		pickle.dump([self.auds_fp, mean, std], f)
		f.close()

		return mean, std

		
