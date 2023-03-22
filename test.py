#All of the needed packages will be imported here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

ps = PorterStemmer()
stopwords = set(stopwords.words('english'))
status = 0.

arxiv = pd.read_csv('ArXiv-10/arxiv100.csv', sep=",")

size = arxiv["abstract"].size
limit = 1
limited_size = int(size * limit)
print("Limited size: ", limited_size)
print("Total size: ", size)

corpus = []
for i in range(limited_size):
    text = arxiv["abstract"][i]
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    
    text = ' '.join([ps.stem(word) for word in text.split() if not word in stopwords])

    current_status = i/limited_size
    if (current_status - status) > 0.01:
        status = current_status
        print(round(status * 100, 1), "%")
    
    corpus.append(text)

cv = CountVectorizer()
X = cv.fit_transform(corpus)
y = arxiv["label"][:limited_size]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


