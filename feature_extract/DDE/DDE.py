import re
import math
import readFasta
import numpy as np
import pandas as pd


def DDE(fastas, **kw):


	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	# 61 categary
	myCodons = {
		'A': 4,
		'C': 2,
		'D': 2,
		'E': 2,
		'F': 2,
		'G': 4,
		'H': 2,
		'I': 3,
		'K': 2,
		'L': 6,
		'M': 1,
		'N': 2,
		'P': 4,
		'Q': 2,
		'R': 6,
		'S': 6,
		'T': 4,
		'V': 4,
		'W': 1,
		'Y': 2
	}

	# 二肽组成：
	encodings = []
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
	header = ['#'] + diPeptides
	encodings.append(header)

	#TM
	myTM = []
	for pair in diPeptides:
		myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])

		code = [name]
		myDC = [0] * 400
		for j in range(len(sequence) - 2 + 1):
			myDC[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = myDC[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1

		# myDC
		if sum(myDC) != 0:
			myDC = [i/sum(myDC) for i in myDC]

		#TV
		myTV = []
		for j in range(len(myTM)):
			myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

		#DDE
		for j in range(len(myDC)):
			myDC[j] = (myDC[j] - myTM[j]) / math.sqrt(myTV[j])

		code = code + myDC
		encodings.append(code)
	return encodings

kw=  {'path': r"raw-glutaryllysine-pos.fasta",'order': 'ACDEFGHIKLMNPQRSTVWY'}
fastas1 = readFasta.readFasta(r"raw-glutaryllysine-pos.fasta")

result=DDE(fastas1, **kw)
data1=np.matrix(result[1:])[:,1:]
data_=pd.DataFrame(data=data1)
data_.to_csv('DDE_row_pos.csv')
