from matplotlib import pyplot as plt
import numpy as np

tp = {'left':False, 'right':False, 'labelleft':False, 'labelbottom':False, 'bottom':False}
def viz_pair(AS, idx, t='', sp_kwangs={}, f_kwangs={}, s_kwangs={}):

	frame, spec, label, indx = AS[idx]

	frame = np.transpose(np.array(frame), axes=[1,2,0])
	spec = np.squeeze(np.array(spec))

	frame -= np.min(frame)
	frame /= np.max(frame)

	spec -= np.min(spec)
	spec /= np.max(spec)

	if 'nrows' not in sp_kwangs.keys():
		sp_kwangs['nrows'] = 1
	if 'ncols' not in sp_kwangs.keys():
		sp_kwangs['ncols'] = 2
	if 'figsize' not in sp_kwangs.keys():
		sp_kwangs['figsize'] = (8,4)

	fig,ax = plt.subplots(**sp_kwangs)
	ax[0].imshow(frame, **f_kwangs)
	ax[1].imshow(spec, **s_kwangs)

	indx = list(indx)
	match_str = 'real pair' if label.item() == 1 else 'fake pair'
	ax[1].set_title('{} [{} {} {} {}]\n{}'.format(t, int(indx[0]),int(indx[1]),int(indx[2]),int(indx[3]), match_str), y=0, pad=7)
	ax[0].tick_params('both', **tp)
	ax[1].tick_params('both', **tp)
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

def viz_clip(AS, idx, t='', sp_kwangs={}, f_kwangs={}, s_kwangs={}):
	if 'nrows' not in sp_kwangs.keys():
		sp_kwangs['nrows'] = 2
	if 'ncols' not in sp_kwangs.keys():
		sp_kwangs['ncols'] = 9
	if 'figsize' not in sp_kwangs.keys():
		sp_kwangs['figsize'] = (18,4)

	fig,ax = plt.subplots(**sp_kwangs)
	ax[1][8].set_title('{}[{}] {}'.format(t, idx, AS.id(idx)), y=0, pad=7)
	for i in range(9):
		frame, spec = AS.view(idx, i)
		frame = np.transpose(np.array(frame), axes=[1,2,0])
		spec = np.squeeze(np.array(spec))
		ax[0][i].imshow(frame, **f_kwangs)
		ax[1][i].imshow(spec, **s_kwangs)
		ax[0][i].tick_params('both', **tp)
		ax[1][i].tick_params('both', **tp)
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

	


