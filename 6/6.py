"""
6. Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier model to
perform this task. Built-in Java classes/API can be used to write the program. Calculate the
accuracy, precision, and recall for your data set.
"""

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn import metrics

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

news = fetch_20newsgroups()
print("All Targets\n", news.target_names)

news_train = fetch_20newsgroups(subset='train', categories=categories, shuffle='true')
news_test = fetch_20newsgroups(subset='test', categories=categories, shuffle='true')
print("Target Names", news_train.target_names)

text_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])
predicted_target = text_clf.fit(news_train["data"], news_train["target"]).predict(news_test["data"])

print("Accuracy", metrics.accuracy_score(news_test["target"], predicted_target))
print(metrics.classification_report(news_test["target"], predicted_target, target_names=news_test.target_names))
print("Confusion Matrix:\n", metrics.confusion_matrix(news_test.target, predicted_target))

"""
Output:

All Targets
 ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
Target Names ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
Accuracy 0.8348868175765646
                        precision    recall  f1-score   support

           alt.atheism       0.97      0.60      0.74       319
         comp.graphics       0.96      0.89      0.92       389
               sci.med       0.97      0.81      0.88       396
soc.religion.christian       0.65      0.99      0.78       398

              accuracy                           0.83      1502
             macro avg       0.89      0.82      0.83      1502
          weighted avg       0.88      0.83      0.84      1502

Confusion Matrix:
 [[192   2   6 119]
 [  2 347   4  36]
 [  2  11 322  61]
 [  2   2   1 393]]
"""
