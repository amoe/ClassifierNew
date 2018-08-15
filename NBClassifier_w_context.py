import re
import pandas as pd
from collections import defaultdict
from heapq import nlargest
import pickle
from Email import Email
from sklearn.model_selection import KFold
from numpy import array
import settings

class NBClassifier:
    
    def __init__(self):
        # The features generated for each line
        self.features = ['containsDear', 'lengthUnder12', 'endsComma', 'containsDashes', 
                         'endsColon', 'containsForwarded', 'inFirst10Perc', 'inLast10Perc',
                         'isSenderEnron', 'prevLineBlank', 'nextLineBlank', 'containsFrom',
                         'containsTo', 'containsDate', 'containsSubject', 'containsDoc', 
                         'containsWpd', 'containsPdf', 'beginsGreater', 'containsUnderscores',
                         'containsNumbers', 'containsAster', 'inAngleBrac', 'inDoubleAngleBrac',
                         'endsFullStop', 'endsExcla', 'containsHi', 'containsHello',
                         'startsDash', 'prevlineG', 'prevlineB', 'prevlineSE', 
                         'prevlineSO', 'prevlineSA', 'prevlineA', 'prevlineTH',
                         'prevlineTG', 'prevlineTB', 'prevlineTSO', 'prevlineTSA',
                         'prevlineNone']
        self.dataMeans = pd.DataFrame()
        # The likelihood of each line class
        self.probClass = defaultdict(int)
        self.lineTypes = ['g', 'b', 'se', 'so', 'sa', 'a', 'th', 'tg', 'tb', 'tso', 'tsa']
    
    def getFeatures(self, email, number, prevClass):
        lineText = email.getLine(int(number)-1)
        containsDear = 1 if 'dear' in lineText.lower() else 0
        lengthUnder12 = 1 if len(lineText) < 12 else 0
        endsComma = 1 if lineText.endswith(',') else 0
        containsDashes = 1 if '----' in lineText else 0
        endsColon = 1 if lineText.endswith(':') else 0
        containsForwarded = 1 if 'forwarded by' in lineText.lower() else 0
        inFirst10Perc = 1 if email.getPosition(number) <= 0.1 else 0
        inLast10Perc = 1 if email.getPosition(number) >= 0.9 else 0
        isSenderEnron = 1 if email.sender.endswith('enron.com') else 0
        prevLineBlank = 0
        if not int(number) == 1:
            prevLineText = email.getLine(int(number)-2)
            if prevLineText.strip() == "":
                prevLineBlank = 1
        nextLineBlank = 0
        if not int(number) == email.getNoLines():
            nextLineText = email.getLine(int(number))
            if nextLineText.strip() == "":
                nextLineBlank = 1
        containsFrom = 1 if 'From:' in lineText else 0
        containsTo = 1 if 'To:' in lineText else 0
        containsDate = 1 if 'Date:' in lineText else 0
        containsSubject = 1 if 'Subject:' in lineText else 0
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
        
        
        prevLineClasses = []
        for lineType in self.lineTypes:
            if prevClass == lineType:
                prevLineClasses.append(1)
            else: prevLineClasses.append(0)
        if prevClass == 'none':
            prevLineClasses.append(1)
        else: prevLineClasses.append(0)
        
        features = list((containsDear, lengthUnder12, endsComma, containsDashes,
                         endsColon, containsForwarded, inFirst10Perc, inLast10Perc,
                         isSenderEnron, prevLineBlank, nextLineBlank, containsFrom,
                         containsTo, containsDate, containsSubject, containsDoc,
                         containsWpd, containsPdf, beginsGreater, containsUnderscores,
                         containsNumbers, containsAster, inAngleBrac, inDoubleAngleBrac,
                         endsFullStop, endsExcla, containsHi, containsHello, startsDash))
        features.extend(prevLineClasses)
        
        return features
        

    def train(self, lineClasses):
#        lineTypes = ['g', 'b', 'se', 'so', 'sa', 'a', 'th', 'tg', 'tb', 'tso', 'tsa']
        
        data = pd.DataFrame()
        dataList = list() # list to change to dataframe
        
        lineIDs = list((lineClasses)) # list of file paths
        
        
        # Build features for each line
        for lineID in lineIDs:
            fp = lineID.split('lineno')[0]
            lineNo = lineID.split('lineno')[1]
            
            if int(lineNo) > 1: # it isn't the first line
                prevClass = lineClasses[fp+'lineno'+str(int(lineNo)-1)] # add class of previous line
            else:
                prevClass = 'none'
                
            email = Email(fp)
            featuresList = [lineClasses[lineID]]
            featuresList.extend(self.getFeatures(email, lineNo, prevClass))
            
#            for lineType in self.lineTypes:
#                if lineType == prevClass:
#                    featuresList.extend([1])
#                else: featuresList.extend([0])
#            if prevClass == 'none':
#                featuresList.extend([1])
#            else: featuresList.extend([0])
            dataList.append(featuresList)
            
                
                
            
        data = pd.DataFrame.from_records(dataList)
        columnList = ['class']
        columnList.extend(self.features)
        data.columns = columnList
        
        # total number of lines in the set
        total_lines = data['class'].count()
        
        # count number of lines in each line class
        typeTotals = defaultdict(int)
        for lineType in self.lineTypes:
            typeTotals[lineType] = data['class'][data['class'] == lineType].count()
            
        # calculate proportion of each line type
        self.probClass = defaultdict(int)
        for lineType in self.lineTypes:
            self.probClass[lineType] = typeTotals[lineType]/total_lines
            
        #print(self.probClass)
            
        # create dataframe with totals for each class/feature combination
        classFeatureSums = data.groupby('class').sum()
        classFeatureSums += 1 # smoothing
        
        dataList = list()
        
        # add columns for inverse of features
        # no inverse for previous line classes
        for index, row in classFeatureSums.iterrows():
            rowList = list()
            rowList.append(index)
            for column in classFeatureSums:
                rowList.append(row[column])
                if not column.startswith('prevline'): # don't add inverse previous lines
                    inverse = (typeTotals[index]+2) - row[column]
                    rowList.append(inverse)
            dataList.append(rowList)
            
        featureTotals = pd.DataFrame.from_records(dataList)
        columnList = ['class']
        for column in classFeatureSums:
            columnList.append(column)
            if not column.startswith('prevline'):
                columnList.append(('not' + column))
        featureTotals.columns = columnList
        featureTotals.set_index('class', inplace=True)
        
        # calculate means for each feature and class
        meanList = list()
        for index, row in featureTotals.iterrows():
            rowList = list()
            rowList.append(index)
            for column in featureTotals:
                rowList.append((row[column])/((typeTotals[index])+2))
            meanList.append(rowList)
            
        self.dataMeans = pd.DataFrame.from_records(meanList)
        columnList = ['class']
        columnList.extend(featureTotals.columns)
        self.dataMeans.columns = columnList
        self.dataMeans.set_index('class', inplace=True)
        
            
    def makePrediction(self, email, lineNo, prevPrediction):
        classProbabilities = {}
        
        testLine = pd.DataFrame()
        
        testLine = pd.DataFrame.from_records([self.getFeatures(email, lineNo, prevPrediction)])
        testLine.columns = self.features
        
        for index1, row1 in self.dataMeans.iterrows(): # for each class in the means table
            classProbabilities[index1] = self.probClass[index1] # add probability of being that class
            for index2, row2 in testLine.iterrows():
                for column in testLine: # for each feature
                    if not column.startswith('prevline'):
                        if row2[column] == 0: # the line doesn't have that feature
                            # multiply by probability of that class not having that feature
                            classProbabilities[index1] = classProbabilities[index1] * row1['not' + column]
                        else: # the line does have that feature
                            # multiply by probability of that class having that feature
                            classProbabilities[index1] = classProbabilities[index1] * row1[column]
                    else:
                        if row2[column] == 1:
                            classProbabilities[index1] = classProbabilities[index1] * row1[column]
                        
        #for key, value in classProbabilities.items():
            #print('"{0}" probability: {1}'.format(key, value))
            
            
        predictedClass = nlargest(1, classProbabilities, key=classProbabilities.get)[0]
        return predictedClass
    
        
      
nb = NBClassifier()
with open(settings.LINE_CLASSES_PATH, 'rb') as f:
    lineClasses = pickle.load(f)
with open(settings.SAMPLE_EMAILS_PATH, 'rb') as f:
    emailsList = pickle.load(f)

emailsArray = array(emailsList)
#train, test = train_test_split(emailsArray, test_size=0.2)
lineList = list((lineClasses))
trainLines = {}
testLines = {}


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
    nb.train(trainLines)
    prevPrediction = 'none'
    predictedClasses = {}
    filepath = 'none'
    for line in testLines:
        if not line.split('lineno')[0] == filepath: # if new filepath doesn't match old filepath
            # this is the first line of a new email
            prevPrediction = 'none'
            filepath = line.split('lineno')[0]
        
        email = Email(line.split('lineno')[0])
        prevPrediction = nb.makePrediction(email, line.split('lineno')[1], prevPrediction)
        predictedClasses[line] = prevPrediction
    
    correct = 0
    for key, value in predictedClasses.items():
        if value == lineClasses[key]:
            correct += 1
    accuracies.append((correct/float(len(testLines)))*100)

overallAccuracy = sum(accuracies)/len(accuracies)
    
print('Overall accuracy: {0}'.format("%.2f" % overallAccuracy))

#for line in lineList:
#    fp = line.split('lineno')[0]
#    if fp in train:
#        trainLines[line] = lineClasses[line]
#    if fp in test:
#        testLines[line] = lineClasses[line]

#nb.train(trainLines)

#predictedClasses = {}

#for email in emails:
#    prevPrediction = 'none'
#    for i in range(1, email.getNoLines()+1):
#        print('\n\n')
#        print(email.getLine(i-1))
#        prevPrediction = nb.makePrediction(email, i, prevPrediction)
#        print('The class is: {0}'.format(prevPrediction))
#        predictedClasses['{0}lineno{1}'.format(email.filepath, i)] = prevPrediction


#for line in testLines:
#    email = Email(line.split('lineno')[0])
#    prevPrediction = nb.makePrediction(email, line.split('lineno')[1], prevPrediction)
#    predictedClasses[line] = prevPrediction

#correct = 0
#for key, value in predictedClasses.items():
#    if value == lineClasses[key]:
#        correct += 1
        
#print('Accuracy: {0}%'.format((correct/float(len(testLines)))*100))






