#This script is used to preprocess a txt novel to a format that can be used in the 'Improved plaintext modelling' part. 
#The preprocessed corpus is stored in add.txt
import os
import re

if(os.path.exists('./a2data/add.txt')):
	os.remove('./a2data/add.txt')

source=open('./a2data/source.txt')

raw=''
for line in source:
	if(not line is '\n'):#remove void lines
		raw=raw+line.strip('\n')
raw=raw.lower()
text=raw.split('.')#Sentence segmentation

states=list()
for i in range(0,26):
	a=ord('a')
	c=chr(i+a)
	states.append(c)
states.extend([' ',',','.','\n'])

f=open('./a2data/add.txt','a')
for line in text:
	x=True
	line=line.strip(' ')#Removing extra space from the beginning and end of each sentence
	if(len(line)<15):#remove very shot lines, which usually are not whole sentences
		x=False
	for char in line:#Removing sentences that contain any other character which is not one of the 29 symbols
		if(not char in states):
			x=False
	if(x==True):
		f.write(line+'.'+'\n')
f.close()