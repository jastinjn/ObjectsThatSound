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


def norm(x):
	x -= np.min(x)
	x /= np.max(x)
	return x

tp = {'left':False, 'right':False, 'labelleft':False, 'labelbottom':False, 'bottom':False}

cfg = get_config('config.ini')
weights_path='C:/Users/Joel Huang/Projects/EECS 442 Project/save_main/AVOLNet100.pt'

size=None
tAS = AudioSet('test', cfg['SAVE_DIR'], VAL_RATIO=cfg['VAL_RATIO'], TEST_RATIO=cfg['TEST_RATIO'], FT_RATIO=cfg['FT_RATIO'], size=size)


model = AVOLNet()
model.load_state_dict(torch.load(weights_path, map_location="cpu"))

print(model.visionNet.net[0].weight.shape)

conv_layer = 0
conv_weights = []
layer_images = []
num_layers = 0
for layer in model.visionNet.net:
	if isinstance(layer, nn.Conv2d):
		num_layers += 1

fig,ax = plt.subplots(nrows=num_layers, ncols=1, figsize=(5, 5*num_layers))
for layer in model.visionNet.net:
	if isinstance(layer, nn.Conv2d):
		
		print(list(layer.weight.shape))
		if conv_layer == 0:

			width = int(np.sqrt(layer.weight.shape[0]))
			img_width = width * 3 + (width-1)
			layer_image = np.transpose(layer.weight.detach().numpy(), axes=[0,2,3,1])

			layer_images.append(layer_image)

			print(list(layer_image.shape))

			layer_use_image = np.reshape(layer_image, (width, width, *tuple(layer.weight.shape[1:])))

			img = np.zeros((img_width,)*2+(3,))
			for i in range(width):
				for j in range(width):
					img[i*4:i*4+3, j*4:j*4+3] = norm(layer_use_image[i,j])

			ax[conv_layer].imshow(img)
			ax[conv_layer].tick_params('both', **tp)

		else:
			height = int(2**(int(np.log2(layer.weight.shape[0])/2)))
			width = int(layer.weight.shape[0]/height)
			print('hw: {} {}'.format(height, width))
			conv_size = 3 + 2*conv_layer
			img_height = height * conv_size + (height-1)
			img_width = width * conv_size + (width-1)

			pcs = conv_size - 2
			prev_images = layer_images[conv_layer-1]
			layer_image = np.zeros((layer.weight.shape[0],conv_size,conv_size,3))
			weights = layer.weight.detach().numpy()
			for C in range(layer.weight.shape[0]):
				for i in range(3):
					for j in range(3):
						layer_image[C, i:i+pcs, j:j+pcs] += \
							np.sum(np.expand_dims(weights[C,:,i,j],axis=[1,2,3]) * prev_images, axis=0)
						#for pC in range(len(prev_images)):
						#	layer_image[C, i:i+pcs, j:j+pcs] += \
						#		weights[C,pC,i,j] * prev_images[pC]

			layer_images.append(layer_image)
			print(list(layer_image.shape))
			layer_use_image = np.reshape(layer_image, (width, height, *tuple(layer_image.shape[1:])))
			
			img = np.zeros((img_width, img_height, 3))
			print('img shape: {}'.format(img.shape))
			cs = conv_size + 1
			for i in range(width):
				for j in range(height):
					img[i*cs:i*cs+conv_size,j*cs:j*cs+conv_size] = norm(layer_use_image[i,j])

			ax[conv_layer].imshow(np.transpose(img, axes=[1,0,2]))
			ax[conv_layer].tick_params('both', **tp)

		conv_layer += 1
		
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0.05)
plt.savefig('weights_plot.jpg', dpi=300)

#def viz_local(local, vid, aud, lab, indx, dir, t):
#	N = len(vid)
#	L = np.array(local)
#	loc = local.reshape(local.shape[0], 1, local.shape[1], local.shape[2])
#	loc = loc.expand(loc.shape[0], 3, loc.shape[2], loc.shape[3])
#	loc = up(loc)
#	loc[:,0,:,:] = 0
#	loc[:,2,:,:] = 0
#	loc *= 3
#	fig,ax = plt.subplots(nrows=2, ncols=N, figsize=(2*N, 4))
#	vid = np.transpose(np.array((vid + loc).detach().cpu()), axes=[0,2,3,1])
#	aud = np.squeeze(np.array(aud.detach().cpu()))
#	for i in range(N):
#		print('\r{}/{} visualisations'.format(i,N-1), end='')
#		maxi = np.argmax(L[i])
#		mini = np.argmin(L[i])
#		vid[i] -= np.min(vid[i])
#		ax[0][i].imshow(vid[i]/np.max(vid[i]))
#		ax[1][i].imshow(aud[i])
#		c = 'black' if np.min(vid[i,100:, 50:-50]) > 0.4 else 'white'
#		#ax[0][i].set_title('{},{} {:.2f}\n{},{} {:.2f}'\
#		#					.format(mini//L.shape[1], mini%L.shape[1], np.min(L[i]),\
#		#							maxi//L.shape[1], maxi%L.shape[1], np.max(L[i])), y=0, pad=7, c=c)
#		ax[0][i].set_title('{:.2f} - {:.2f}'\
#							.format(np.min(L[i]), np.max(L[i])), y=0, pad=7, c=c)
#		ax[1][i].set_title('{} {:.0f}:{:.0f} {:.0f}:{:.0f}'\
#			.format(match_str[int(lab[i].item())], \
#			indx[i,0].item(), indx[i,2].item(), indx[i,1].item(), indx[i,3].item()), c='white',\
#			y=0, pad=7)
#		ax[0][i].tick_params('both', **tp)
#		ax[1][i].tick_params('both', **tp)

#	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
#	plt.savefig('{}/viz_model{}.jpg'.format(dir, t), dpi=300)
#	plt.close(fig)


