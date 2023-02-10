import matplotlib.pyplot as plt
import numpy as np

main_fp = 'D:/JH/OneDrive - Umich/6 - 2022 Fall/EECS 442/Project/training_log1.txt'
lesl_fp = 'D:/JH/OneDrive - Umich/6 - 2022 Fall/EECS 442/Project/training_log2 filled.txt'

#main_fp = 'C:/Users/Joel Huang/Projects/EECS 442 Project/save_small_main/training_log.txt'
#lesl_fp = 'C:/Users/Joel Huang/Projects/EECS 442 Project/save_small_less_layers/training_log.txt'
#lesc_fp = 'C:/Users/Joel Huang/Projects/EECS 442 Project/save_small_less_chn/training_log.txt'


def movingAve(x):
	w=7
	y = np.zeros_like(x)
	for i in range(len(x)):
		y[i] = np.mean(x[max(i-w,0):min(i+w+1,len(x)-1)])
	return y

def getLossHistory(FILENAME):
	f = open(FILENAME, 'r')
	tloss = []
	tacc = []
	vloss = []
	vacc = []

	key_tloss = 'training loss '
	key_tacc  = 'train accuracy '
	key_vloss = 'validation loss '
	key_vacc  = 'validation accuracy '
	while True:
		line = f.readline()
		if not line:
			break

		i_tloss = line.find(key_tloss)
		i_tacc  = line.find(key_tacc )
		i_vloss = line.find(key_vloss)
		i_vacc  = line.find(key_vacc )
		tloss.append(float(line[i_tloss+len(key_tloss):line.find(',', i_tloss)]))
		tacc.append( float(line[i_tacc +len(key_tacc) :line.find(',', i_tacc )]))	
		vloss.append(float(line[i_vloss+len(key_vloss):line.find(',', i_vloss)]))
		vacc.append( float(line[i_vacc +len(key_vacc) :line.find(',', i_vacc )]))	
	f.close()
	return np.array(tloss), np.array(tacc), np.array(vloss), np.array(vacc)


main_tloss, main_tacc, main_vloss, main_vacc = getLossHistory(main_fp)
lesl_tloss, lesl_tacc, lesl_vloss, lesl_vacc = getLossHistory(lesl_fp)
#lesc_tloss, lesc_tacc, lesc_vloss, lesc_vacc = getLossHistory(lesc_fp)
mainX = np.arange(len(main_tloss))
leslX = np.arange(len(lesl_tloss))
#lescX = np.arange(len(lesc_tloss))

#tloss_max = np.max(lesl_tloss)
#vloss_max = np.max(lesl_vloss)
tloss_max = np.max(main_tloss)
vloss_max = np.max(main_vloss)/2
main_vloss[main_vloss > vloss_max] = np.nan
fig,ax = plt.subplots(1, figsize=(5,5))

#ax.plot(mainX, movingAve(main_tloss)/tloss_max, color='red' , label='training (full model)'   , alpha=0.5)
#ax.plot(mainX, movingAve(main_vloss)/vloss_max, color='blue', label='validation (full model)' , alpha=0.5)
#ax.plot(leslX, movingAve(lesl_tloss)/tloss_max, color='red' , label='training (less layers)'  , alpha=0.5, linestyle='dashed')
#ax.plot(leslX, movingAve(lesl_vloss)/vloss_max, color='blue', label='validation (less layers)', alpha=0.5, linestyle='dashed')
#ax.plot(lescX, movingAve(lesc_tloss)/tloss_max, color='red' , label='training (less channels)'  , alpha=0.5, linestyle='dotted')
#ax.plot(lescX, movingAve(lesc_vloss)/vloss_max, color='blue', label='validation (less channels)', alpha=0.5, linestyle='dotted')

#ax.plot(mainX, main_tloss/tloss_max, color='red' , alpha=0.15)
#ax.plot(mainX, main_vloss/vloss_max, color='blue', alpha=0.15)
#ax.plot(leslX, lesl_tloss/tloss_max, color='red' , alpha=0.15, linestyle='dashed')
#ax.plot(leslX, lesl_vloss/vloss_max, color='blue', alpha=0.15, linestyle='dashed')
#ax.plot(lescX, lesc_tloss/tloss_max, color='red' , alpha=0.15, linestyle='dotted')
#ax.plot(lescX, lesc_vloss/vloss_max, color='blue', alpha=0.15, linestyle='dotted')


ax.plot(mainX, main_tloss/tloss_max, color='red' , label='training',   alpha=0.5)
ax.plot(mainX, main_vloss/vloss_max, color='blue', label='validation', alpha=0.5)
ax.plot(leslX, lesl_tloss/tloss_max, color='red' , label='training (augmented dataset)'  )
ax.plot(leslX, lesl_vloss/vloss_max, color='blue', label='validation (augmented dataset)')
#ax.scatter([np.argmin(main_vloss), np.nanargmin(lesl_vloss), np.nanargmin(lesc_vloss)],
#		   [np.min(main_vloss)/vloss_max, np.nanmin(lesl_vloss)/vloss_max, np.nanmin(lesc_vloss)/vloss_max],
#		   marker='o', s=50, alpha=0.5, color='green', label='optimal loss')
ax.scatter([np.argmin(main_vloss), np.nanargmin(lesl_vloss)],
		   [np.min(main_vloss)/vloss_max, np.nanmin(lesl_vloss)/vloss_max],
		   marker='o', s=50, alpha=0.5, color='green', label='optimal loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Normalized Loss')
#print(np.argmin(aug_vloss))
ax.legend(frameon=False, labelspacing=0.2)
plt.savefig('Loss.jpg', dpi=300)

fig,ax = plt.subplots(1, figsize=(5,5))
#ax.plot(mainX, movingAve(main_tacc), color='red' , label='training (full model)'   , alpha=0.5)
#ax.plot(mainX, movingAve(main_vacc), color='blue', label='validation (full model)' , alpha=0.5)
#ax.plot(leslX, movingAve(lesl_tacc), color='red' , label='training (less layers)'  , alpha=0.5, linestyle='dashed')
#ax.plot(leslX, movingAve(lesl_vacc), color='blue', label='validation (less layers)', alpha=0.5, linestyle='dashed')
#ax.plot(lescX, movingAve(lesc_tacc), color='red' , label='training (less channels)'  , alpha=0.5, linestyle='dotted')
#ax.plot(lescX, movingAve(lesc_vacc), color='blue', label='validation (less channels)', alpha=0.5, linestyle='dotted')

#ax.plot(mainX, main_tacc, color='red' , alpha=0.15)
#ax.plot(mainX, main_vacc, color='blue', alpha=0.15)
#ax.plot(leslX, lesl_tacc, color='red' , alpha=0.15, linestyle='dashed')
#ax.plot(leslX, lesl_vacc, color='blue', alpha=0.15, linestyle='dashed')
#ax.plot(lescX, lesc_tacc, color='red' , alpha=0.15, linestyle='dotted')
#ax.plot(lescX, lesc_vacc, color='blue', alpha=0.15, linestyle='dotted')


ax.plot(mainX, main_tacc, color='red' , label='training',   alpha=0.5)
ax.plot(mainX, main_vacc, color='blue', label='validation', alpha=0.5)
ax.plot(leslX, lesl_tacc, color='red' , label='training (augmented dataset)'  )
ax.plot(leslX, lesl_vacc, color='blue', label='validation (augmented dataset)')

#ax.scatter([np.argmax(main_vacc), np.argmax(lesl_vacc), np.argmax(lesc_vacc)],
#		   [np.max(main_vacc), np.max(lesl_vacc), np.max(lesc_vacc)],
#		   marker='o', s=50, alpha=0.5, color='green', label='optimal accuracy')

ax.scatter([np.argmax(main_vacc), np.argmax(lesl_vacc)],
		   [np.max(main_vacc), np.max(lesl_vacc)],
		   marker='o', s=50, alpha=0.5, color='green', label='optimal accuracy')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
#print(np.argmax(aug_vacc))
ax.legend(frameon=False, labelspacing=0.2, loc='upper right')
plt.savefig('Acc.jpg', dpi=300)



#ax.plot(augX, aug_tacc                   , color='red' )
#ax.plot(augX, aug_vacc                   , color='blue')
plt.show()






	