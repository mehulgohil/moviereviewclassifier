from nltk.corpus import movie_reviews
from random import shuffle
from nltk import FreqDist
from nltk.corpus import stopwords
import string
from nltk import NaiveBayesClassifier
from nltk import classify
import pickle
from nltk.tokenize import word_tokenize


positive_review_file = movie_reviews.fileids('pos')[0]


#word list,category
#(['plot', ':', 'two', 'teen', 'couples', 'go', ...], 'neg')
documents=[]
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fileid), category))
shuffle(documents)


all_words = [word.lower() for word in movie_reviews.words()]


#stopwords
stopwords_english = stopwords.words('english')


all_words_clean = []
for word in all_words:
    if word not in stopwords_english and word not in string.punctuation:
        all_words_clean.append(word)


all_words_frequency = FreqDist(all_words_clean)


#common words with its frequency
most_common_words = all_words_frequency.most_common(2000)


#most common words only
word_features = [item[0] for item in most_common_words]


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
for (doc, category) in documents:
    feature_set.append((document_features(doc), category))



test_set = feature_set[:400]
train_set = feature_set[400:]


#classifier = NaiveBayesClassifier.train(train_set)


#load pickle in classifier
classifier_f=open("Naivebayes.pickle" , "rb")
classifier=pickle.load(classifier_f)
classifier_f.close()


accuracy = classify.accuracy(classifier, test_set)
print (accuracy*100)
print (classifier.show_most_informative_features(5))


#Save baive byes in pickle
#save_classifier=open("Naivebayes.pickle","wb")
#pickle.dump(classifier,save_classifier)
#save_classifier.close()

"""
#custom input
custom_review = "I hated the film. It was a disaster. Poor direction, bad acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = document_features(custom_review_tokens)
print (classifier.classify(custom_review_set))

#propability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result)
print (prob_result.max())
print (prob_result.prob("neg"))
print (prob_result.prob("pos"))
"""