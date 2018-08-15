from sklearn.feature_extraction.text import CountVectorizer
from Email import Email
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from numpy import array
import re, string, pickle

class Classifier:
    
    def __init__(self):
        with open('../lineClasses.txt', 'rb') as f:
            self.lineClasses = pickle.load(f)
        with open('../sampleEmails.txt', 'rb') as f:
            self.emailsList = pickle.load(f)
        self.model = linear_model.LogisticRegression(C=1e5)
        self.overallAccuracy = 0
        self.words = []
        
    def getFeatures(self, email, number, words):
        lineText = email.getLine(number)
        lineFeatures = []
        for word in words:
            if word in lineText:
                lineFeatures.append(1)
            else: lineFeatures.append(0)
            
        lengthUnder12 = 1 if len(lineText) < 12 else 0
        endsComma = 1 if lineText.endswith(',') else 0
        containsDashes = 1 if '-----' in lineText else 0
        endsColon = 1 if lineText.endswith(':') else 0
        inFirst10Perc = 1 if email.getPosition(number) <= 0.1 else 0
        inLast10Perc = 1 if email.getPosition(number) >= 0.9 else 0
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
        beginsGreater = 1 if lineText.startswith('>') else 0
        containsUnderscores = 1 if '____' in lineText else 0
        containsNumbers = 1 if any(char.isdigit() for char in lineText) else 0
        containsAster = 1 if '****' in lineText else 0
        inAngleBrac = 1 if re.match('<.*?>', lineText) else 0
        inDoubleAngleBrac = 1 if re.match('<<.*?>>', lineText) else 0
        endsFullStop = 1 if lineText.endswith('.') else 0
        endsExcla = 1 if lineText.endswith('!') else 0
        startsDash = 1 if lineText.strip().startswith('-') else 0
        isLineBlank = 1 if lineText.strip() == '' else 0
        lengthUnder20 = 1 if len(lineText) < 20 else 0
        under3Words = 1 if len(lineText.split()) < 3 else 0
        endsPunct = 1 if len(lineText) > 0 and lineText[-1] in '.?-:;!,' else 0
        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        containsPunct = 1 if count(lineText, string.punctuation) > 0 else 0
        containsAt = 1 if '@' in lineText else 0
        lengthOver50 = 1 if len(lineText) > 50 else 0
        containsForwardSlash = 1 if '/' in lineText else 0
        startsCapLetter = 0
        if len(lineText) > 0:
            startsCapLetter = 1 if lineText[0].isupper() else 0
            
        lineFeatures.extend([lengthUnder12, endsComma, containsDashes, endsColon, inFirst10Perc,
                         inLast10Perc, prevLineBlank, nextLineBlank, beginsGreater,
                         containsUnderscores, containsNumbers, containsAster, inAngleBrac,
                         inDoubleAngleBrac, endsFullStop, endsExcla, startsDash,
                         isLineBlank, lengthUnder20, under3Words, endsPunct, containsPunct,
                         containsAt, lengthOver50, containsForwardSlash, startsCapLetter])
    
        return lineFeatures
    
    def trainModel(self):
        kf = KFold(5, True, 1)
        lineList = list((self.lineClasses))
        trainLines = {}
        testLines = {}
#        y_true = []
#        y_pred = []
        accuracies = []
        emailsArray = array(self.emailsList)
        
#        classWords = defaultdict(list)
#        for lineID, lineType in trainLines.items():
#            filepath = lineID.split('lineno')[0]
#            number = lineID.split('lineno')[1]
#            email = Email(filepath)
#            lineText = email.getLine(number)
#            classWords[lineType].append(lineText)
#            
#        for value in classWords.values():
#            for word in value:
#                if not word in self.words:
#                    self.words.append(word)
        
        for train_index, test_index in kf.split(emailsArray):
            trainFPs = emailsArray[train_index]
            for line in lineList:
                fp = line.split('lineno')[0]
                if fp in trainFPs:
                    trainLines[line] = self.lineClasses[line]
                else:
                    testLines[line] = self.lineClasses[line]
            lineIDs = list((testLines))
            X = list()
            Y = list()
            
            classWords = defaultdict(list)
            for lineID, lineType in trainLines.items():
                filepath = lineID.split('lineno')[0]
                number = lineID.split('lineno')[1]
                email = Email(filepath)
                lineText = email.getLine(number)
                classWords[lineType].append(lineText)
                
            words = []
            for value in classWords.values():
                for word in value:
                    if not word in words:
                        words.append(word)
                        
            self.words = words
                        
            for lineID in lineIDs:
                filepath = lineID.split('lineno')[0]
                number = lineID.split('lineno')[1]
                email = Email(filepath)
#                X.append(self.getFeatures(email, number, self.words))
                X.append(self.getFeatures(email, number, words))
                Y.append(self.lineClasses[lineID])
            self.model.fit(X, Y)
            
            predictedClasses = {}
            for line in testLines:
                email = Email(line.split('lineno')[0])
#                lineFeatures = self.getFeatures(email, line.split('lineno')[1], self.words)
                lineFeatures = self.getFeatures(email, line.split('lineno')[1], words)
                prediction = self.model.predict([lineFeatures])
                predictedClasses[line] = prediction
                
            correct = 0
            
            for key, value in predictedClasses.items():
#                y_true.append(lineClasses[key])
#                y_pred.append(value)
                if value == self.lineClasses[key]:
                    correct += 1
                else:
                    # assume all lines misclassified as separators are correct -
                    # this means blank lines within e.g. a body do not lower the accuracy
                    if value == 'se':
                        correct += 1
                    
            accuracies.append((correct/float(len(testLines)))*100)
            print((correct/float(len(testLines)))*100)
            
        self.overallAccuracy = sum(accuracies)/len(accuracies)
    
    def predictEmail(self, filepath):
        email = Email(filepath)
        predictions = []
        for i in range(1, email.getNoLines()+1):
            lineText = email.getLine(i)
            lineFeatures = self.getFeatures(email, i, self.words)
            prediction = self.model.predict([lineFeatures])
            predictions.append(prediction)
#            print('{0} === {1}'.format(lineText, prediction))
        return predictions
            