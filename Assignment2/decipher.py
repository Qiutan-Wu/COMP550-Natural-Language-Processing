import nltk.tag
import os.path
import re
import argparse
from nltk.probability import LaplaceProbDist,ConditionalFreqDist,ConditionalProbDist,MLEProbDist

parser=argparse.ArgumentParser()
parser.add_argument('cipher_folder',help="path of the folder containing cipher text and plain text")
parser.add_argument("-lm", "--lm",help="use additional text to improve modelling",action="store_true")
parser.add_argument('-laplace','--laplace',help="turn on laplace smoothing",action="store_true")

args=parser.parse_args()

cipher_folder=args.cipher_folder
estimator=None

if(args.laplace):#if -laplace is turned on, the estimator in tagger training should be Laplace Smoothing. if not, MLE is used
	estimator=LaplaceProbDist
	print('Laplace Smoothing is on')		
	
# cipher_folder='./a2data/cipher1'
	
states=list()
for i in range(0,26):
	a=ord('a')
	c=chr(i+a)
	states.append(c)
states.extend([' ',',','.'])

def TransitionsGenerate(AddCorpus,train_p,tagger,estimator):#recalculate the transition matrix using train plain text + additional corpus

	if estimator is None:
		estimator = lambda fdist, bins: MLEProbDist(fdist)

	data=train_p
	data.extend(AddCorpus)
	print(type(data))
	for s in data:
		s=list(s)

	N =len(tagger._states)

	transitions = ConditionalFreqDist()
	for sentence in data:
		lasts=None
		sentence=list(sentence.strip('\n'))
		for character in sentence:
			state=character
			if not lasts is None:
				transitions[lasts][state] += 1
			lasts=state
	A = ConditionalProbDist(transitions, estimator, N)
	return A


train_p=list()
train_c=list()
test_p=list()
test_c=list()

f1=open(os.path.join(cipher_folder,'train_cipher.txt'))
f2=open(os.path.join(cipher_folder,'train_plain.txt'))
f3=open(os.path.join(cipher_folder,'test_cipher.txt'))
f4=open(os.path.join(cipher_folder,'test_plain.txt'))

for line in f1:
	train_c.append(line)
for line in f2:
	train_p.append(line)
for line in f3:
	test_c.append(line)
for line in f4:
	test_p.append(line)

f1.close()
f2.close()
f3.close()
f4.close()

train_data=list()
for i in range(0,len(train_c)):
	zipped=zip(list(train_c[i].strip('\n')),list(train_p[i].strip('\n')))
	train_data.append(list(zipped))

trainer=nltk.tag.hmm.HiddenMarkovModelTrainer(states=states)

tagger = trainer.train_supervised(train_data,estimator=estimator)


if(args.lm):#if -lm is turned on, replace the trantions matrix in tagger by the recalculated one
	print('Improved plaintext modelling is on')
	f5=open('./a2data/add.txt')
	AddCorpus=list()
	for line in f5:
		AddCorpus.append(line)
	ImprovedTransitions=TransitionsGenerate(AddCorpus,train_p,tagger,estimator=estimator)
	tagger._transitions=ImprovedTransitions
	f5.close()
	
#############test, print result and accuracy######################
for line in test_c:
	line=list(line.strip('\n'))
	tags=tagger.tag(line)
	c=''
	p=''
	for tag in tags:
		c=c+tag[0]
		p=p+tag[1]
	print('cipher: ',c)
	print('plain:  ',p)
	print('\n')

test_data=list()
for i in range(0,len(test_c)):
	zipped=zip(list(test_c[i].strip('\n')),list(test_p[i].strip('\n')))
	test_data.append(list(zipped))
tagger.test(test_data)


