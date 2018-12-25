import librosa
from tqdm import tqdm
from multiprocessing import Process
import math
import numpy as np

mode='test'

with open('./Dataset/'+mode+'_filenames.txt') as f:
	filenames = ['./Dataset/'+mode+'Norm/'+x[:-1] for x in f ]

def convert(items):
	for fn in tqdm(items):
		raw,sr=librosa.load(fn)
		outn=fn.replace('/'+mode+'Norm','/'+mode+'RAW').replace('.flac','.npy')
		if(raw.shape[0] >= 220500):
			np.save(outn,raw[:220500])
		else:
			print(outn,raw.shape)

n_threads = 16
sl=[]
chunk_size = int(math.ceil(float(len(filenames))/n_threads))
for n in range(n_threads):
	start=n*chunk_size
	chunk=filenames[start:start+chunk_size]
	print(len(chunk))
	sl.append(chunk)

threads = []
for n in range(n_threads):
	chunk = sl[n]
	p = Process(target=convert, args=(chunk,))
	p.start()
	threads.append(p)

for t in threads:
	t.join()

	
