import numpy as np
import matplotlib.pyplot as plt
import torch
from os.path import exists
from config import get_config
from torch.utils.data import DataLoader
from torch import nn
from dataset import AudioSet
from yt_seg import YT_SEG_Converter

from main_model import AVOLNet

import pickle

tp = {'left':False, 'right':False, 'labelleft':False, 'labelbottom':False, 'bottom':False}
match_str = ['fake pair','real pair']
up = nn.Upsample(scale_factor=16)

cfg = get_config('config.ini')
weights_path='C:/Users/Joel Huang/Projects/EECS 442 Project/save_main/AVOLNet100.pt'

size=None
tAS = AudioSet('test', cfg['SAVE_DIR'], VAL_RATIO=cfg['VAL_RATIO'], TEST_RATIO=cfg['TEST_RATIO'], FT_RATIO=cfg['FT_RATIO'], size=size)

pickleFP = 'YT_SEG.pickle'
if not exists(pickleFP):
	YTS = YT_SEG_Converter(tAS, 'D:/JH/OneDrive - Umich/6 - 2022 Fall/EECS 442/Project/Code/audio_files')
	f = open(pickleFP, 'wb')
	pickle.dump((YTS), f)
	f.close()
else:
	f = open(pickleFP, 'rb')
	YTS = pickle.load(f)
	f.close()

def viz_local(local, vid, aud, lab, indx, dir, t):
	N = len(vid)
	L = np.array(local)
	loc = local.reshape(local.shape[0], 1, local.shape[1], local.shape[2])
	loc = loc.expand(loc.shape[0], 3, loc.shape[2], loc.shape[3])
	loc = up(loc)
	loc[:,0,:,:] = 0
	loc[:,2,:,:] = 0
	loc *= 3
	fig,ax = plt.subplots(nrows=2, ncols=N, figsize=(2*N, 4))
	vid = np.transpose(np.array((vid + loc).detach().cpu()), axes=[0,2,3,1])
	aud = np.squeeze(np.array(aud.detach().cpu()))
	for i in range(N):
		print('\r{}/{} visualisations'.format(i,N-1), end='')
		maxi = np.argmax(L[i])
		mini = np.argmin(L[i])
		vid[i] -= np.min(vid[i])
		ax[0][i].imshow(vid[i]/np.max(vid[i]))
		ax[1][i].imshow(aud[i])
		c = 'black' if np.min(vid[i,100:, 50:-50]) > 0.4 else 'white'
		#ax[0][i].set_title('{},{} {:.2f}\n{},{} {:.2f}'\
		#					.format(mini//L.shape[1], mini%L.shape[1], np.min(L[i]),\
		#							maxi//L.shape[1], maxi%L.shape[1], np.max(L[i])), y=0, pad=7, c=c)
		ax[0][i].set_title('{:.2f} - {:.2f}'\
							.format(np.min(L[i]), np.max(L[i])), y=0, pad=7, c=c)
		ax[1][i].set_title('{} {:.0f}:{:.0f} {:.0f}:{:.0f}'\
			.format(match_str[int(lab[i].item())], \
			indx[i,0].item(), indx[i,2].item(), indx[i,1].item(), indx[i,3].item()), c='white',\
			y=0, pad=7)
		ax[0][i].tick_params('both', **tp)
		ax[1][i].tick_params('both', **tp)

	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
	plt.savefig('{}/viz_model{}.jpg'.format(dir, t), dpi=300)
	plt.close(fig)



def collate_data(idx, i, jdx, j):
	vid, aud, lab, indx = None, None, None, None
	for I in range(len(i)):
		V, A, L, ind = tAS.__getitem__(idx[I], str(i[I]), jdx[I], str(j[I]))
		if vid is None:
			vid = torch.zeros((len(i),) + V.shape)
			aud = torch.zeros((len(i),) + A.shape)
			lab = torch.zeros((len(i),) + L.shape)
			indx= torch.zeros((len(i),) + ind.shape)
		vid[I] = V
		aud[I] = A
		lab[I] = L
		indx[I] = ind
	return vid, aud, lab, indx

model = AVOLNet()
model.load_state_dict(torch.load(weights_path, map_location="cpu"))


idx = [2,3,5,6,6,9,12,13,19,21,25,26,27,28,29,35,37,38,42,44,55,71,71,82,93,98,113,114,115,119,126,141,163,186,187,215,273]
i =	  [0,0,4,0,8,3,3 ,6 ,3 ,0 ,0 ,1 ,7 ,2 ,0 ,6 ,0 ,0 ,5 ,5 ,4 ,4 ,6 ,8 ,4 ,0 ,6,  4,  0  ,8  ,0  ,0  ,0  ,8  ,0  ,0  ,7]
vid, aud, lab, indx = collate_data(idx, i, idx, i)
out,local = model(vid, aud)
viz_local(local.detach(), vid.detach(), aud.detach(), lab.detach(), indx.detach(), 'audio_files', ' true pairs')

#idx = np.arange(260, 280)
#i =	  np.random.randint(0, 8, idx.shape)
#idx = [262,]
#i   = [4  ,]
#vid, aud, lab, indx = collate_data(idx, i, idx, i)
#out,local = model(vid, aud)
#viz_local(local.detach(), vid.detach(), aud.detach(), lab.detach(), indx.detach(), 'audio_files', ' true pairs 2')


for I in range(len(idx)):
	YTS.get_segment(idx[I], i[I])

idx = [71,71 ,6,9 ,55,71,71,71,38,163,273,21 ,119,262,268,13,19,42]
i =	  [4 ,4  ,8,3 ,4 ,4 ,4 ,4 ,0 ,0  ,7  ,0  ,8  ,4  ,3  ,6 ,3 ,5 ]
jdx = [44,141,9,28,0 ,82,6 ,27,28,186,186,119,71 ,44 ,141,19,13,13]
j =	  [5 ,0  ,3,2 ,0 ,8 ,0 ,7 ,2 ,8  ,8  ,8  ,6  ,5  ,0  ,3 ,6 ,6 ]
vid, aud, lab, indx = collate_data(idx, i, jdx, j)
out,local = model(vid, aud)
viz_local(local.detach(), vid.detach(), aud.detach(), lab.detach(), indx.detach(), 'audio_files', ' false pairs')





