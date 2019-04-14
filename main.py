import nltk
from nltk.corpus import movie_reviews
from random import shuffle
from nltk import FreqDist
from nltk.corpus import stopwords
import string
from nltk import NaiveBayesClassifier
from nltk import classify
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


#positive_review_file = movie_reviews.fileids('pos')[0]


#word list,category
#(['plot', ':', 'two', 'teen', 'couples', 'go', ...], 'neg')


"""
documents=[]
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fileid), category))
shuffle(documents)
"""

#all_words = [word.lower() for word in movie_reviews.words()]
#print (all_words[:10])

#stopwords
#stopwords_english = stopwords.words('english')


"""
all_words_clean = []
for word in all_words:
    if word not in stopwords_english and word not in string.punctuation:
        all_words_clean.append(word)
"""

"""#no noun words
no_noun=[]
for word in all_words_clean:
    word = nltk.word_tokenize(word)
    tagged= nltk.pos_tag(word)
    if tagged[-1] != 'NNP' and tagged[-1] != 'NNPS':
        no_noun.append(word)


all_words_clean_1 = []
for word in all_words_clean:
    if word not in no_noun:
        all_words_clean_1.append(word)
"""

#all_words_frequency = FreqDist(all_words_clean)


#common words with its frequency
#most_common_words = all_words_frequency.most_common(2000)


#most common words only
#word_features = [item[0] for item in most_common_words]


#wordfeatures_new=open("wordfeatures.pickle","wb")
#pickle.dump(word_features,wordfeatures_new)
#wordfeatures_new.close()


wordfeatures_new=open("wordfeatures.pickle","rb")
word_features=pickle.load(wordfeatures_new)


"""
{'contains(waste)': False, 'contains(lot)': False, 'contains(rent)': False,
 'contains(black)': False, 'contains(rated)': False, 'contains(potential)': False,
"""


def document_features(document):
    document_words = set(document)  # "set" function will remove repeated/duplicate
    features = {}       #Dictionary
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


feature_set = []
"""
for (doc, category) in documents:
    feature_set.append((document_features(doc), category))
"""

"""
featureset_new=open("featureset.pickle","wb")
pickle.dump(feature_set,featureset_new)
featureset_new.close()
"""

featureset_new=open("featureset.pickle","rb")
feature_set=pickle.load(featureset_new)

test_set = feature_set[:400]
print(test_set[1])
train_set = feature_set[400:]


#classifier = NaiveBayesClassifier.train(train_set)

#Save baive byes in pickle
#save_classifier=open("Naivebayes.pickle","wb")
#pickle.dump(classifier,save_classifier)
#save_classifier.close()

#load pickle in classifier
classifier_f=open("Naivebayes.pickle" , "rb")
classifier=pickle.load(classifier_f)
classifier_f.close()

print (classifier.show_most_informative_features(10))

accuracy = classify.accuracy(classifier, test_set)
print ("NaiveBayes accuracy percent:",accuracy*100)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, test_set)*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(train_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, test_set)*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(train_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(train_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(train_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, test_set))*100)


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier)


print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, test_set))*100)


custom_review = "It was a wonderful and amazing movie. I loved it. Best direction, good acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = document_features(custom_review_tokens)

#print(voted_classifier.classify(custom_review_set))  # Output: neg
# Positive review is classified as negative
# We need to improve our feature set for more accurate prediction
print("Classification:", voted_classifier.classify(custom_review_set), "Confidence %:",voted_classifier.confidence(custom_review_set)*100)
