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
    cc = 0
    mask = np.ones(len(dataMat[1]), dtype=bool)
    for col in range(0,len(dataMat[1])):
        if max(dataMat[:,cc]) == 0:
            mask[cc] = False
        else:
            dataMat[:,cc] = dataMat[:,cc]/max(dataMat[:,cc])
            if np.var(dataMat[:,cc])<=0.015:
                mask[cc]=False
        cc += 1
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
	for c2 in range(1,colCount-3):
		trainData[count-1][c2] = data[count][c2]
	dt = data[count][0].rsplit(" ")
	trainData[count-1][0] = int(dt[1].rsplit(":")[0])


#################################### kfold ################################
K = rowCount/2
# for iter in range(0, rowCount/K) :
	# startIndex = iter*K
	# if((iter+1)*K
print K
fakeTest = matCopy(trainData, 0,K)
fakeTestTargetC = targetC[0:K]
fakeTestTargetR = targetR[0:K]
fakeTestTarget = map(add,fakeTestTargetR,fakeTestTargetC)

fakeTrain = trainData[K:rowCount]
fakeTrainTargetC = targetC[K:rowCount]
fakeTrainTargetR = targetR[K:rowCount]


################################### Load actual test ################################
# reader = csv.reader(open(testFilePath),delimiter=",")
# testd = list(reader)
# colCount = len(testd[1])
# rowCount = len(testd)
# testTargetID = []
# testData = np.ones((rowCount-1,colCount))
# for count in range(1,rowCount):
	# for c2 in range(1,colCount):
		# testData[count-1][c2] = testd[count][c2]
	# dt = testd[count][0].rsplit(" ")
	# testData[count-1][0] = int(dt[1].rsplit(":")[0])
	# testTargetID.append(testd[count][0])

################################### Process #####################################
# colToSkip = normalize(trainData)
# maskedTrD = trainData[:,colToSkip]

# normalize(testData)
# maskedTD = testData[:,colToSkip]


#################################### Classfiy ####################################
#NCclf = NearestCentroid()
#NCclf.fit(maskedTrD,target)
#NCprediction = NCclf.predict(maskedTD)

# SVMclf = svm.SVC()
# SVMclf.fit(maskedTrD,target)
# SVMprediction = SVMclf.predict(maskedTD)

# GNBclf = GaussianNB()
# GNBprediction = GNBclf.fit(maskedTrD,target).predict(maskedTD)

# Tclf = tree.DecisionTreeClassifier()
# Tclf.fit(maskedTrD,target)
# Tprediction = Tclf.predict(maskedTD)

# SGDclf = SGDClassifier(loss="log")
# SGDclf.fit(maskedTrD,target)
# SGDprediction = SGDclf.predict(maskedTD)
# SGDprob = SGDclf.predict_log_proba(maskedTD)

# KNclf = KNeighborsClassifier(n_neighbors=50)
# KNclf.fit(maskedTrD,target)
# KNprediction = KNclf.predict(maskedTD)

# RCFclf = RandomForestClassifier(n_estimators=300,max_features=7)
# RCFprediction = RCFclf.fit(maskedTrD,target).predict(maskedTD)

RCFclf = RandomForestClassifier(n_estimators=10,max_features=9)
RCFpredictionC = RCFclf.fit(fakeTrain,fakeTrainTargetC).predict(fakeTest)
RCFpredictionR = RCFclf.fit(fakeTrain,fakeTrainTargetR).predict(fakeTest)
RCFprediction = map(add, RCFpredictionR, RCFpredictionC)
print RCFprediction[0]

# RCFclf = RandomForestClassifier(n_estimators=10,max_features=9)
# RCFprediction = RCFclf.fit(trainData,target).predict(testData)

# ETclf = ExtraTreesClassifier(n_estimators=200)
# ETprediction = ETclf.fit(maskedTrD,target).predict(maskedTD)

#GDBclf = GradientBoostingClassifier(n_estimators = 100)
#GDBprediction = GDBclf.fit(maskedTrD,target).predict(maskedTD)
##################################### Evaluate ####################################

numCorr = 0
ii = 0
total = 0
for ii in range(0,len(fakeTestTargetC)) :
	total += math.pow(math.log(RCFprediction[ii]+1)-math.log(fakeTestTarget[ii]+1),2)
accuracy = math.sqrt(total/len(fakeTestTargetC))
print("RCF's Accuracy :",accuracy)


############################## write output ###################################
# with open(outPath,'wb') as csvFile:
   # writer = csv.writer(csvFile,delimiter = ',')
   # writer.writerow(['datetime','count'])
   # for ii in range(0,len(fakeTestTargetC)):
       # writer.writerow([testTargetID[ii], RCFprediction[ii]])

############################## Confusion Matrix ###############################
# import matplotlib.pyplot as plt

# cm = confusion_matrix(testTarget,ETprediction)
# print(cm)
# plt.matshow(cm)
# plt.colorbar()
# plt.ylabel('Actual class')
# plt.xlabel('Predicted class')
# plt.savefig('confusion_mat.png', format='png')

















