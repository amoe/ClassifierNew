from Email import Email
import re
import string
from sklearn.naive_bayes import BernoulliNB
import pickle
from sklearn.model_selection import KFold
from numpy import array
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import operator
from heapq import nlargest
import settings

features = ['containsDear', 'lengthUnder12', 'endsComma', 'containsDashes', 
            'endsColon', 'containsForwarded', 'inFirst10Perc', 'inLast10Perc',
            'isSenderEnron', 'prevLineBlank', 'nextLineBlank', 'containsFrom',
            'containsTo', 'containsDate', 'containsSubject', 'containsDoc', 
            'containsWpd', 'containsPdf', 'beginsGreater', 'containsUnderscores',
            'containsNumbers', 'containsAster', 'inAngleBrac', 'inDoubleAngleBrac',
            'endsFullStop', 'endsExcla', 'containsHi', 'containsHello', 'startsDash',
            'isLineBlank', 'lengthUnder20', 'under3Words', 'endsPunct', 'containsPunct',
            'containsThanks', 'containsBest', 'containsSincerely', 'containsRegards',
            'containsAt', 'containsCC', 'lengthOver50', 'containsSent', 'containsForwardSlash',
            'startsCapLetter']

def getFeatures(email, number):
    lineText = email.getLine(int(number)-1)
    containsDear = 1 if 'dear' in lineText.lower() else 0
    lengthUnder12 = 1 if len(lineText) < 12 else 0
    endsComma = 1 if lineText.endswith(',') else 0
    containsDashes = 1 if '-----' in lineText else 0
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
    isLineBlank = 1 if lineText.strip() == '' else 0
    lengthUnder20 = 1 if len(lineText) < 20 else 0
    under3Words = 1 if len(lineText.split()) < 3 else 0
    endsPunct = 1 if len(lineText) > 0 and lineText[-1] in '.?-:;!,' else 0
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    containsPunct = 1 if count(lineText, string.punctuation) > 0 else 0
    containsThanks = 1 if 'thanks' in lineText.lower() else 0
    containsBest = 1 if 'best' in lineText.lower() else 0
    containsSincerely = 1 if 'sincerely' in lineText.lower() else 0
    containsRegards = 1 if 'regards' in lineText.lower() else 0
    containsAt = 1 if '@' in lineText else 0
    containsCC = 1 if 'cc:' in lineText.lower() else 0
    lengthOver50 = 1 if len(lineText) > 50 else 0
    containsSent = 1 if 'sent:' in lineText.lower() else 0
    containsForwardSlash = 1 if '/' in lineText else 0
    startsCapLetter = 0
    if len(lineText) > 0:
        startsCapLetter = 1 if lineText[0].isupper() else 0
    
    return list((containsDear, lengthUnder12, endsComma, containsDashes, endsColon,
                 containsForwarded, inFirst10Perc, inLast10Perc, isSenderEnron,
                 prevLineBlank, nextLineBlank, containsFrom, containsTo, containsDate,
                 containsSubject, containsDoc, containsWpd, containsPdf, beginsGreater,
                 containsUnderscores, containsNumbers, containsAster, inAngleBrac,
                 inDoubleAngleBrac, endsFullStop, endsExcla, containsHi, containsHello,
                 startsDash, isLineBlank, lengthUnder20, under3Words, endsPunct,
                 containsPunct, containsThanks, containsBest, containsSincerely,
                 containsRegards, containsAt, containsCC, lengthOver50, containsSent,
                 containsForwardSlash, startsCapLetter))
    
with open(settings.LINE_CLASSES_PATH, 'rb') as f:
    lineClasses = pickle.load(f)
with open(settings.SAMPLE_EMAILS_PATH, 'rb') as f:
    emailsList = pickle.load(f)
    
labels=['a', 'b', 'g', 'sa', 'se', 'so', 'tb', 'tg', 'th', 'tsa', 'tso']

emailsArray = array(emailsList)


def trainTestModel(model, emailsArray):
    kf = KFold(5, True, 1)
    lineList = list((lineClasses))
    trainLines = {}
    testLines = {}
    y_true = []
    y_pred = []
    trainAccuracy = []
    accuracies = []
    importantFeatures = defaultdict(int)
    
    # k fold testing on all data
    for train_index, test_index in kf.split(emailsArray):
        trainFPs = emailsArray[train_index]
        for line in lineList:
            fp = line.split('lineno')[0]
            if fp in trainFPs:
                trainLines[line] = lineClasses[line]
            else:
                testLines[line] = lineClasses[line]
        lineIDs = list((testLines))
        X = list()
        Y = list()
        # create data and target value
        for lineID in lineIDs:
            fp = lineID.split('lineno')[0]
            lineNo = lineID.split('lineno')[1]
            email = Email(fp)
            X.append(getFeatures(email, lineNo))
            Y.append(lineClasses[lineID])
        model.fit(X, Y)
        
        # test the model using the data used for training
        trainPredictedClasses = {}
        for line in trainLines:
            email = Email(line.split('lineno')[0])
            lineFeatures = getFeatures(email, line.split('lineno')[1])
            prediction = model.predict([lineFeatures])
            trainPredictedClasses[line] = prediction
            
        correct = 0
        for key, value in trainPredictedClasses.items():
            if value == lineClasses[key]:
                correct += 1
        trainAccuracy.append((correct/float(len(trainLines)))*100)
        
        # test the model using unseen data
        predictedClasses = {}
        for line in testLines:
            email = Email(line.split('lineno')[0])
            testFeatures = getFeatures(email, line.split('lineno')[1])
#            print(model.predict_proba([testFeatures])) 
            prediction = model.predict([testFeatures])
            predictedClasses[line] = prediction
            
        correct = 0
        wrongClass = defaultdict(int)
        secondBest = defaultdict(int)
        cl = 'tso'
        
#        coeff = model.coef_
        classes = ['a', 'b', 'g', 'se', 'so', 'tb', 'tg', 'th', 'tsa', 'tso']
#        classToWeights = dict(zip(classes, coeff))
        
        # count number of correct classifications
#        filepath = 'none'
        for key, value in predictedClasses.items():
            # print list of line classifications for each email
#            if not key.split('lineno')[0] == filepath:
#                filepath = key.split('lineno')[0]
#                print('\n\n\nNEW EMAIL\n')
#            print(value)
            y_true.append(lineClasses[key])
            y_pred.append(value)
            if value == lineClasses[key]:
                correct += 1
            else:
                if lineClasses[key] == cl and value == 'tb': # the real class was cl and the predicted class was this
                    
                    em = Email(key.split('lineno')[0])
                    
                    # get weights of features for the predicted class
#                    weights = classToWeights[value[0]]
#                    featuresToWeights = dict(zip(features, weights))
#                    # get the 10 most important features for the predicted class
#                    impFeatures = nlargest(10, featuresToWeights, key=featuresToWeights.get)
#                    lineFeatures = dict(zip(features, getFeatures(em, key.split('lineno')[1])))
#                    lineImpFeatures = []
#                    
#                    for feature in impFeatures:
#                        if lineFeatures[feature] == 1:
#                            lineImpFeatures.append(feature)
#                            importantFeatures[feature] += 1
                    
                    
                    
                    # calculate second most likely class
#                    probabilities = model.predict_proba([getFeatures(em, key.split('lineno')[1])])
#                    i=0
#                    classes = ['a', 'b', 'g', 'se', 'so', 'tb', 'tg', 'th', 'tsa', 'tso']
#                    classProbs = dict(zip(classes, probabilities[0]))
#                    twoLargest = nlargest(2, classProbs, key=classProbs.get)
#                    wrongClass[twoLargest[0]] += 1
#                    secondBest[twoLargest[1]] += 1
#                    print('{0} classified as {1}'.format(em.getLine(int(key.split('lineno')[1])-1), value))
#                    print('\n\n\nfeatures this {0} line has that are important features for {1}:\n{2}'.format(lineClasses[key], value[0], lineImpFeatures))
                
        accuracies.append((correct/float(len(testLines)))*100)
        
#    print(importantFeatures)
        
    overallTrainAccuracy = sum(trainAccuracy)/len(trainAccuracy)
    overallAccuracy = sum(accuracies)/len(accuracies)
#    sortedWrongClass = sorted(wrongClass.items(), key=operator.itemgetter(1))
#    sortedSecondBest = sorted(secondBest.items(), key=operator.itemgetter(1))
    
#    print('\nincorrect {0} predictions:\n{1}\n\nsecond best predictions for those:\n{2}'.format(cl, sortedWrongClass, sortedSecondBest))
    
    cm = confusion_matrix(y_true, y_pred, labels)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    modelType = ((str(type(model)).split('.')[-1])[:-2])
    ax.set_title('Confusion matrix for {0}'.format(modelType))
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()
    print('Overall training accuracy: {0}'.format("%.2f" % overallTrainAccuracy))
    print('Overall test accuracy: {0}'.format("%.2f" % overallAccuracy))
#    coeff = model.coef_
#    
#    return coeff


    

# -------- naive bayes --------

bnb = BernoulliNB()
trainTestModel(bnb, emailsArray)




# -------- knn --------

knn = KNeighborsClassifier()
trainTestModel(knn, emailsArray)





# -------- logistic regression --------

regr = linear_model.LogisticRegression(C=1e5)
trainTestModel(regr, emailsArray)

# print weights for each class/feature pair
#i=0
#for label in labels:
#    if not label == 'sa':
#        j=0
#        for feature in features:
#            if abs(coeff[i][j]) >= 1:
#                print('feature {0} class {1}: {2}'.format(feature, label, coeff[i][j]))
#            j+=1
#        i+=1




# -------- svm --------

svc = svm.SVC(kernel='linear')
trainTestModel(svc, emailsArray)


