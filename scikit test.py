from Email import Email
import re
from sklearn.naive_bayes import BernoulliNB
import pickle
from sklearn.model_selection import KFold
from numpy import array
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

features = ['containsDear', 'lengthUnder12', 'endsComma', 'containsDashes', 
            'endsColon', 'containsForwarded', 'inFirst10Perc', 'inLast10Perc',
            'isSenderEnron', 'prevLineBlank', 'nextLineBlank', 'containsFrom',
            'containsTo', 'containsDate', 'containsSubject', 'containsDoc', 
            'containsWpd', 'containsPdf', 'beginsGreater', 'containsUnderscores',
            'containsNumbers', 'containsAster', 'inAngleBrac', 'inDoubleAngleBrac',
            'endsFullStop', 'endsExcla', 'containsHi', 'containsHello', 'startsDash']

def getFeatures(email, number):
    lineText = email.getLine(int(number)-1)
    containsDear = 1 if 'dear' in lineText.lower() else 0
    lengthUnder12 = 1 if len(lineText) < 12 else 0
    endsComma = 1 if lineText.endswith(',') else 0
    containsDashes = 1 if '----' in lineText else 0
    endsColon = 1 if lineText.endswith(',') else 0
    containsForwarded = 1 if 'forwarded by' in lineText.lower() else 0
    inFirst10Perc = 1 if email.getPosition(number) <= 0.1 else 0
    inLast10Perc = 1 if email.getPosition(number) >= 0.9 else 0
    isSenderEnron = 1 if email.sender.endswith('enron.com') else 0
    prevLineBlank = 0
    if not int(number) == 1:
        prevLineText = email.getLine(int(number)-2)
        if prevLineText.strip() == '':
            prevLineBlank = 1
    nextLineBlank = 0
    if not int(number) == email.getNoLines():
        nextLineText = email.getLine(int(number))
        if nextLineText.strip() == '':
            nextLineBlank = 1
    containsFrom = 1 if 'from:' in lineText.lower() else 0
    containsTo = 1 if 'to:' in lineText.lower() else 0
    containsDate = 1 if 'date:' in lineText.lower() else 0
    containsSubject = 1 if 'subject:' in lineText.lower() else 0
    containsDoc = 1 if '.doc' in lineText.lower() else 0
    containsWpd = 1 if '.wpd' in lineText.lower() else 0
    containsPdf = 1 if '.pdf' in lineText.lower() else 0
    beginsGreater = 1 if lineText.startswith('>') else 0
    containsUnderscores = 1 if '____' in lineText else 0
    containsNumbers = 1 if any(char.isdigit() for char in lineText) else 0
    containsAster = 1 if '****' in lineText else 0
    inAngleBrac = 1 if re.match('<.*?>', lineText) else 0
    inDoubleAngleBrac = 1 if re.match('<<.*?>>', lineText) else 0
    endsFullStop = 1 if lineText.endswith('.') else 0
    endsExcla = 1 if lineText.endswith('!') else 0
    containsHi = 1 if 'hi' in lineText.lower() else 0
    containsHello = 1 if 'hello' in lineText.lower() else 0
    startsDash = 1 if lineText.strip().startswith('-') else 0
    
    return list((containsDear, lengthUnder12, endsComma, containsDashes, endsColon,
                 containsForwarded, inFirst10Perc, inLast10Perc, isSenderEnron,
                 prevLineBlank, nextLineBlank, containsFrom, containsTo, containsDate,
                 containsSubject, containsDoc, containsWpd, containsPdf, beginsGreater,
                 containsUnderscores, containsNumbers, containsAster, inAngleBrac,
                 inDoubleAngleBrac, endsFullStop, endsExcla, containsHi, containsHello,
                 startsDash))
    
with open('lineClasses.txt', 'rb') as f:
    lineClasses = pickle.load(f)
with open('sampleEmails.txt', 'rb') as f:
    emailsList = pickle.load(f)
    
labels=['a', 'b', 'g', 'sa', 'se', 'so', 'tb', 'tg', 'th', 'tsa', 'tso']


    

# -------- naive bayes --------

bnb = BernoulliNB()
emailsArray = array(emailsList)
lineList = list((lineClasses))
trainLines = {}
testLines = {}
y_true = []
y_pred = []

kf = KFold(5, True, 1)
accuracies = []
for train_index, test_index in kf.split(emailsArray):
    trainFPs = emailsArray[train_index]
    testFPs = emailsArray[test_index]
    for line in lineList:
        fp = line.split('lineno')[0]
        if fp in trainFPs:
            trainLines[line] = lineClasses[line]
        else:
            testLines[line] = lineClasses[line]
    lineIDs = list((testLines))
    X = list()
    Y = list()
    for lineID in lineIDs:
        fp = lineID.split('lineno')[0]
        lineNo = lineID.split('lineno')[1]
        email = Email(fp)
        X.append(getFeatures(email, lineNo))
        Y.append(lineClasses[lineID])
    bnb.fit(X, Y)
    
    predictedClasses = {}
    filepath = 'none'
    for line in testLines:
        if not line.split('lineno')[0] == filepath:
            filepath = line.split('lineno')[0]
            
        email = Email(line.split('lineno')[0])
        testFeatures = getFeatures(email, line.split('lineno')[1])
        prediction = bnb.predict([testFeatures])
        predictedClasses[line] = prediction
        
    correct = 0
    for key, value in predictedClasses.items():
        y_true.append(lineClasses[key])
        y_pred.append(value)
        if value == lineClasses[key]:
            correct += 1
    
    
    accuracies.append((correct/float(len(testLines)))*100)
    
overallAccuracy = sum(accuracies)/len(accuracies)

# show confusion matrix for all folds combined
cm = confusion_matrix(y_true, y_pred, labels)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion matrix for Naive Bayes classification')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()
print('Overall accuracy: {0}'.format("%.2f" % overallAccuracy))




# -------- knn --------

knn = KNeighborsClassifier()
emailsArray = array(emailsList)
lineList = list((lineClasses))
trainLines = {}
testLines = {}
y_true = []
y_pred = []

kf = KFold(5, True, 1)
accuracies = []
for train_index, test_index in kf.split(emailsArray):
    trainFPs = emailsArray[train_index]
    testFPs = emailsArray[test_index]
    for line in lineList:
        fp = line.split('lineno')[0]
        if fp in trainFPs:
            trainLines[line] = lineClasses[line]
        else:
            testLines[line] = lineClasses[line]
    lineIDs = list((testLines))
    X = list()
    Y = list()
    for lineID in lineIDs:
        fp = lineID.split('lineno')[0]
        lineNo = lineID.split('lineno')[1]
        email = Email(fp)
        X.append(getFeatures(email, lineNo))
        Y.append(lineClasses[lineID])
    knn.fit(X, Y)
    
    predictedClasses = {}
    filepath = 'none'
    for line in testLines:
        if not line.split('lineno')[0] == filepath:
            filepath = line.split('lineno')[0]
            
        email = Email(line.split('lineno')[0])
        testFeatures = getFeatures(email, line.split('lineno')[1])
        prediction = knn.predict([testFeatures])
        predictedClasses[line] = prediction

    correct = 0
    for key, value in predictedClasses.items():
        y_true.append(lineClasses[key])
        y_pred.append(value)
        if value == lineClasses[key]:
            correct += 1
    
    
    accuracies.append((correct/float(len(testLines)))*100)
    
overallAccuracy = sum(accuracies)/len(accuracies)

# show confusion matrix for all folds combined
cm = confusion_matrix(y_true, y_pred, labels)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion matrix for K Nearest Neighbours classification')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()
print('Overall accuracy: {0}'.format("%.2f" % overallAccuracy))





# -------- logistic regression --------

regr = linear_model.LogisticRegression(C=1e5)
emailsArray = array(emailsList)
lineList = list((lineClasses))
trainLines = {}
testLines = {}
y_true = []
y_pred = []

kf = KFold(5, True, 6)
accuracies = []
for train_index, test_index in kf.split(emailsArray):
    trainFPs = emailsArray[train_index]
    testFPs = emailsArray[test_index]
    for line in lineList:
        fp = line.split('lineno')[0]
        if fp in trainFPs:
            trainLines[line] = lineClasses[line]
        else:
            testLines[line] = lineClasses[line]
    lineIDs = list((testLines))
    X = list()
    Y = list()
    for lineID in lineIDs:
        fp = lineID.split('lineno')[0]
        lineNo = lineID.split('lineno')[1]
        email = Email(fp)
        X.append(getFeatures(email, lineNo))
        Y.append(lineClasses[lineID])
    regr.fit(X, Y)
    
    predictedClasses = {}
    filepath = 'none'
    for line in testLines:
        if not line.split('lineno')[0] == filepath:
            filepath = line.split('lineno')[0]
            
        email = Email(line.split('lineno')[0])
        testFeatures = getFeatures(email, line.split('lineno')[1])
        prediction = regr.predict([testFeatures])
        predictedClasses[line] = prediction
        
    correct = 0
    for key, value in predictedClasses.items():
        y_true.append(lineClasses[key])
        y_pred.append(value)
        if value == lineClasses[key]:
            correct += 1
    
    
    accuracies.append((correct/float(len(testLines)))*100)
    
overallAccuracy = sum(accuracies)/len(accuracies)

# show confusion matrix for all folds combined
cm = confusion_matrix(y_true, y_pred, labels)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion matrix for Logistic Regression classification')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()
print('Overall accuracy: {0}'.format("%.2f" % overallAccuracy))




# -------- svm --------

svc = svm.SVC(kernel='linear')
emailsArray = array(emailsList)
lineList = list((lineClasses))
trainLines = {}
testLines = {}
i = 1
y_true = []
y_pred = []
kf = KFold(5, True, 1)
accuracies = []
for train_index, test_index in kf.split(emailsArray):
    trainFPs = emailsArray[train_index]
    testFPs = emailsArray[test_index]
    for line in lineList:
        fp = line.split('lineno')[0]
        if fp in trainFPs:
            trainLines[line] = lineClasses[line]
        else:
            testLines[line] = lineClasses[line]
    lineIDs = list((testLines))
    X = list()
    Y = list()
    for lineID in lineIDs:
        fp = lineID.split('lineno')[0]
        lineNo = lineID.split('lineno')[1]
        email = Email(fp)
        X.append(getFeatures(email, lineNo))
        Y.append(lineClasses[lineID])
    svc.fit(X, Y)
    
    predictedClasses = {}
    filepath = 'none'
    for line in testLines:
        if not line.split('lineno')[0] == filepath:
            filepath = line.split('lineno')[0]
            
        email = Email(line.split('lineno')[0])
        testFeatures = getFeatures(email, line.split('lineno')[1])
        prediction = svc.predict([testFeatures])
        predictedClasses[line] = prediction
        
    correct = 0
    for key, value in predictedClasses.items():
        y_true.append(lineClasses[key])
        y_pred.append(value)
        if value == lineClasses[key]:
            correct += 1
    
    
    accuracies.append((correct/float(len(testLines)))*100)
    
overallAccuracy = sum(accuracies)/len(accuracies)

# show confusion matrix for all folds combined
cm = confusion_matrix(y_true, y_pred, labels)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion matrix for SVM classification')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()
print('Overall accuracy: {0}'.format("%.2f" % overallAccuracy))


