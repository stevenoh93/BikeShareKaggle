#!D:\School\Python34

import numpy as np
import csv
import math
from operator import add
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# Normalize. Returns a mask
def normalize(dataMat):
    mask = np.ones(len(dataMat[1]), dtype=bool)
    for col in range(0,len(dataMat[1])):
        if max(dataMat[:,col]) == 0:
            mask[col] = False
        else:
            dataMat[:,col] = dataMat[:,col]/max(dataMat[:,col])
            if np.var(dataMat[:,col])<=0.015:
                mask[col]=False
    return mask
	
def matCopy(mat, rows, rowt) :
	mat2 = []
	for row in range(rows,rowt) :
		curRow=[];
		for col in range(0, len(mat[row])) :
			curRow.append(mat[row][col])
		mat2.append(curRow)
	return mat2


trainFilePath = "train.csv"
testFilePath = "test.csv"
outPath = "out.csv"
#################################### Load data ####################################
reader = csv.reader(open(trainFilePath),delimiter=",")
data = list(reader)
colCount = len(data[1])
rowCount = len(data)
targetC = np.arange(rowCount-1)
targetR = np.arange(rowCount-1)
colNames = data[0]
trainData = np.ones((rowCount-1,colCount-3))
for count in range(1,rowCount):
	targetC[count-1] = int(data[count][colCount-3])
	targetR[count-1] = int(data[count][colCount-2])
	for c2 in range(2,colCount-3):
		trainData[count-1][c2] = data[count][c2]
	dt = data[count][0].rsplit(" ")
	monthnday = dt[0].split("-")
	trainData[count-1][0] = int(dt[1].rsplit(":")[0])
	trainData[count-1][1] = int(monthnday[0]) + float(monthnday[1])/30


#################################### kfold ################################
K = rowCount/2
step = 2
# for iter in range(0, rowCount/K) :
	# startIndex = iter*K
	# if((iter+1)*K
# fakeTest = matCopy(trainData, 0,K)
# fakeTestTargetC = targetC[0:K]
# fakeTestTargetR = targetR[0:K]
# fakeTestTarget = map(add,fakeTestTargetR,fakeTestTargetC)

# fakeTrain = trainData[K:rowCount]
# fakeTrainTargetC = targetC[K:rowCount]
# fakeTrainTargetR = targetR[K:rowCount]

fakeTest = trainData[0:rowCount:step]
fakeTestTargetC = targetC[0:rowCount:step]
fakeTestTargetR = targetR[0:rowCount:step]
fakeTestTarget = map(add,fakeTestTargetR,fakeTestTargetC)

fakeTrain = trainData[1:rowCount:step]
fakeTrainTargetC = targetC[1:rowCount:step]
fakeTrainTargetR = targetR[1:rowCount:step]


################################### Load actual test ################################
reader = csv.reader(open(testFilePath),delimiter=",")
testd = list(reader)
colCount = len(testd[1])
rowCount = len(testd)
testTargetID = []
testData = np.ones((rowCount-1,colCount))
for count in range(1,rowCount):
	for c2 in range(2,colCount):
		testData[count-1][c2] = testd[count][c2]
	dt = testd[count][0].rsplit(" ")
	monthnday = dt[0].split("-")
	testData[count-1][0] = int(dt[1].rsplit(":")[0])
	testData[count-1][1] = int(monthnday[0]) + float(monthnday[1])/30
	testTargetID.append(testd[count][0])

################################### Process #####################################
normalize(fakeTrain)
normalize(fakeTest)

# normalize(testData)
# testData = testData

#################################### Classfiy ####################################
#NCclf = NearestCentroid()
#NCclf.fit(maskedTrD,target)
#NCprediction = NCclf.predict(maskedTD)

# SVMclf = svm.SVC()
# SVMclf.fit(maskedTrD,target)
# SVMprediction = SVMclf.predict(maskedTD)

# GNBclf = GaussianNB()
# GNBprediction = GNBclf.fit(maskedTrD,target).predict(maskedTD)



# SGDclf = SGDClassifier(loss="log")
# SGDclf.fit(maskedTrD,target)
# SGDprediction = SGDclf.predict(maskedTD)
# SGDprob = SGDclf.predict_log_proba(maskedTD)

# KNclf = KNeighborsClassifier(n_neighbors=50)
# KNclf.fit(maskedTrD,target)
# KNprediction = KNclf.predict(maskedTD)

#GDBclf = GradientBoostingClassifier(n_estimators = 100)
#GDBprediction = GDBclf.fit(maskedTrD,target).predict(maskedTD)

RCFclf = RandomForestClassifier(n_estimators=20)
RCFpredictionC = RCFclf.fit(fakeTrain,fakeTrainTargetC).predict(fakeTest)
RCFpredictionR = RCFclf.fit(fakeTrain,fakeTrainTargetR).predict(fakeTest)
RCFprediction = map(add, RCFpredictionR, RCFpredictionC)

Tclf = tree.DecisionTreeClassifier()
TclfpredictionC = Tclf.fit(fakeTrain,fakeTrainTargetC).predict(fakeTest)
TclfpredictionR = Tclf.fit(fakeTrain,fakeTrainTargetR).predict(fakeTest)
Tclfprediction = map(add,TclfpredictionR,TclfpredictionC)

ETclf = ExtraTreesClassifier(n_estimators=4)
ETclfpredictionC = ETclf.fit(fakeTrain,fakeTrainTargetC).predict(fakeTest)
ETclfpredictionR = ETclf.fit(fakeTrain,fakeTrainTargetR).predict(fakeTest)
ETclfprediction = map(add,ETclfpredictionR,ETclfpredictionC)

##################################### Actual Classify ####################################

# RCFclf = RandomForestClassifier(n_estimators=10,max_features=9)
# prediction = RCFclf.fit(trainData,target).predict(testData)

# ETclf = ExtraTreesClassifier(n_estimators=4)
# predictionC = ETclf.fit(trainData,targetC).predict(testData)
# predictionR = ETclf.fit(trainData,targetR).predict(testData)
# prediction = map(add,predictionC,predictionR)

Tclf = tree.DecisionTreeClassifier()
TclfpredictionC = Tclf.fit(trainData,targetC).predict(testData)
TclfpredictionR = Tclf.fit(trainData,targetR).predict(testData)
prediction = map(add,TclfpredictionR,TclfpredictionC)

##################################### Evaluate ####################################

numCorr = 0
ii = 0
totalR = 0
totalT = 0
totalE = 0
for ii in range(0,len(fakeTestTargetC)) :
	totalR += math.pow(math.log(RCFprediction[ii]+1)-math.log(fakeTestTarget[ii]+1),2)
	totalT += math.pow(math.log(Tclfprediction[ii]+1)-math.log(fakeTestTarget[ii]+1),2)
	totalE += math.pow(math.log(ETclfprediction[ii]+1)-math.log(fakeTestTarget[ii]+1),2)
accuracyR = math.sqrt(totalR/len(fakeTestTargetC))
accuracyT = math.sqrt(totalT/len(fakeTestTargetC))
accuracyE = math.sqrt(totalE/len(fakeTestTargetC))
print("RCF's Accuracy :",accuracyR)
print("ET's Accuracy :",accuracyE)
print("T's Accuracy :",accuracyT)


############################## write output ###################################
# with open(outPath,'wb') as csvFile:
   # writer = csv.writer(csvFile,delimiter = ',')
   # writer.writerow(['datetime','count'])
   # for ii in range(0,len(prediction)):
       # writer.writerow([testTargetID[ii], prediction[ii]])

############################## Confusion Matrix ###############################
# import matplotlib.pyplot as plt

# cm = confusion_matrix(testTarget,ETprediction)
# print(cm)
# plt.matshow(cm)
# plt.colorbar()
# plt.ylabel('Actual class')
# plt.xlabel('Predicted class')
# plt.savefig('confusion_mat.png', format='png')

















