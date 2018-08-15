from sklearn.feature_extraction.text import CountVectorizer
from Email import Email
import pickle
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from numpy import array

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

with open('lineClasses.pkl', 'rb') as f:
    lineClasses = pickle.load(f)
with open('sampleEmails.pkl', 'rb') as f:
    emailsList = pickle.load(f)
    


#for key, value in classWords.items():
#    if not key == 'se':
#        print('\n\n' + key + ':')
#        topWords = get_top_n_words(value, 20)
#        print(topWords)


        
def getFeatures(email, number, features):
    lineText = email.getLine(int(number)-1)
    lineFeatures = []
    for feature in features:
        if feature in lineText:
            lineFeatures.append(1)
        else: lineFeatures.append(0)
    return lineFeatures

def trainTestModel(model, emailsArray):
    kf = KFold(5, True, 1)
    lineList = list((lineClasses))
    trainLines = {}
    testLines = {}
    y_true = []
    y_pred = []
    trainAccuracy = []
    accuracies = []
    
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
        
        
        
        classWords = defaultdict(list)
        for lineID, lineType in trainLines.items():
            filepath = lineID.split('lineno')[0]
            number = lineID.split('lineno')[1]
            email = Email(filepath)
            lineText = email.getLine(int(number)-1)
            classWords[lineType].append(lineText)
        
        topClassWords = defaultdict(list)
        for key, value in classWords.items():
            if not key == 'se':
                topClassWords[key] = get_top_n_words(value, 20)
                
#        print(topClassWords)
        
        # create list of words as features
#        features = []
#        for value in topClassWords.values(): # for each list of words
#            for word in value:
#                if not word[0] in features: # if it hasn't already been added
#                    features.append(word[0])
                    
        features = []
        for value in classWords.values():
            for word in value:
                if not word in features:
                    features.append(word)
                    
        
        for lineID in lineIDs:
            fp = lineID.split('lineno')[0]
            lineNo = lineID.split('lineno')[1]
            email = Email(fp)
            X.append(getFeatures(email, lineNo, features))
            Y.append(lineClasses[lineID])
        model.fit(X, Y)
        
        trainPredictedClasses = {}
        for line in trainLines:
            email = Email(line.split('lineno')[0])
            lineFeatures = getFeatures(email, line.split('lineno')[1], features)
            prediction = model.predict([lineFeatures])
            trainPredictedClasses[line] = prediction
            
        correct = 0
        for key, value in trainPredictedClasses.items():
            if value == lineClasses[key]:
                correct += 1
        trainAccuracy.append((correct/float(len(trainLines)))*100)
        
        predictedClasses = {}
        for line in testLines:
            email = Email(line.split('lineno')[0])
            testFeatures = getFeatures(email, line.split('lineno')[1], features)
            prediction = model.predict([testFeatures])
            predictedClasses[line] = prediction
            
        correct = 0
        
        for key, value in predictedClasses.items():
            y_true.append(lineClasses[key])
            y_pred.append(value)
            if value == lineClasses[key]:
                correct += 1
                
        accuracies.append((correct/float(len(testLines)))*100)
        
    overallTrainAccuracy = sum(trainAccuracy)/len(trainAccuracy)
    overallAccuracy = sum(accuracies)/len(accuracies)
    
    labels = ['a', 'b', 'g', 'sa', 'se', 'so', 'tb', 'tg', 'th', 'tsa', 'tso']
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
    
emailsArray = array(emailsList)
regr = linear_model.LogisticRegression(C=1e5)
trainTestModel(regr, emailsArray)
    