# lolch-classifier-new

Requires Python 3.

Email.py -- contains a class representing a given email from a given filepath.

Every top-level module imports email.

sampleEmails and lineClasses are in pickle format.  Use 
`python3 -m pickle <file>` to inspect them.

lineClasses contains a giant dictionary mapping 

You can find the actual path of the mail defined in the `Email` class.

Dependencies:

`seaborn` (or `python3-seaborn` in debian) `0.8.0`
`sklearn` 0.19.1

## Data set 

You can find the data set at the following link:

https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

This is the canonical version of the dataset as of this writing (2018-08-15).

In `settings.py` you can find the location of the data set.

## Notes

Running `bag_of_words.py` will exercise a model by coercing the training set into
lists _X_,_Y_.  This uses the `sklearn.linear_model.LogisticRegression` ML
approach.  It will do the whole lot: training the model, testing it, and
plotting the results.

NBClassifier - lolch's hand-implementation of naive bayes classifier

'Context' means adding a 'context window' that incorporates the result of the
classification of the previous line into the current classification.  Sort of
a 1-element recurrence.

'Other Features' refers to the features that are extractible by the `getFeatures`
method.

`scikit_test` and `scikit_with_context` are the result of porting the code
to various Scikit models.  Tried items were `sklearn.naive_bayes.BernouilliNB`,
`sklearn.neighbors.KNeighborsClassifier`, and the one with better performance,
the `sklearn.linear_model.LogisticRegression`.

`Classifier` is a ported and cleaned up version of the basic algorithm from
`bag_of_words_and_other_features.py`.

The class names are defined as such:

    # g = greeting
    # b = body
    # se = section separator
    # so = signoff
    # sa = signature
    # a = attachment
    # th = thread header
    # tg = greeting within thread
    # tb = body within thread
    # tso = signoff within thread
    # tsa = signature within thread
