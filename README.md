# lolch-classifier-new

Requires Python 3.

Email.py -- contains a class representing a given email from a given filepath.

Every top-level module imports email.

sampleEmails and lineClasses are in pickle format.  Use 
`python3 -m pickle <file>` to inspect them.

lineClasses contains a giant dictionary mapping 

Bag of words is the simplest model.

You can find the actual path of the mail defined in the `Email` class.

Dependencies:

`seaborn` (or `python3-seaborn` in debian) `0.8.0`

## Data set 

You can find the data set at the following link:

https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

This is the canonical version of the dataset as of this writing (2018-08-15).

