import librosa
from tqdm import tqdm
from multiprocessing import Process
import math
import numpy as np

mode='test'

with open('./Dataset/'+mode+'_filenames.txt') as f:
	filenames = ['./Dataset/'+mode+'Norm/'+x[:-1] for x in f ]

#print (filenames[0],filenames[len(filenames)-1],len(filenames))

n_mfcc=13

def convert(items):
	for fn in tqdm(items):
		y,sr=librosa.load(fn)
		mfcc=librosa.feature.mfcc(y=y,sr=sr,hop_length=512,n_mfcc=n_mfcc)
		outn=fn.replace('/'+mode+'Norm','/'+mode+'MFCC'+str(n_mfcc)).replace('.flac','.npy')
		np.save(outn,mfcc)
		if(mfcc.shape != (n_mfcc,431)):
			print(outn,mfcc.shape)

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

	
