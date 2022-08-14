import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score
import nltk
import re
from nltk.stem import WordNetLemmatizer



def preprocessing():
    test_csv = pd.read_csv('Test.csv', sep=',', header=0)
    train_csv = pd.read_csv('Train.csv', sep=',', header=0)
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    nltk.download('stopwords')
    train_X_non = train_csv['text']
    train_y = train_csv['label']
    test_X_non = test_csv['text']
    test_y = test_csv['label']

    def clear_text(text):
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
        review = ' '.join(review)
        return review

    train_X = train_X_non.apply(lambda x: clear_text(x))
    print(train_X)

    test_X = test_X_non.apply(lambda x: clear_text(x))
    return train_X, test_X, train_y, test_y


def Vectorize():
    train_X, test_X, train_y, test_y = preprocessing()
    # tf idf
    tf_idf = TfidfVectorizer()
    X_train_tf = tf_idf.fit_transform(train_X)
    X_train_tf = tf_idf.transform(train_X)
    X_test_tf = tf_idf.transform(test_X)
    print(X_test_tf)
    return X_train_tf, X_test_tf, train_y, test_y


def LRegression():
    X_train_tf, X_test_tf, train_y, test_y = Vectorize()
    mdl = LogisticRegression().fit(X_train_tf, train_y)

    train_preds, test_preds = mdl.predict_proba(X_train_tf)[:, 1], mdl.predict_proba(X_test_tf)[:, 1]

    train_predictions = mdl.predict(X_train_tf)
    accuracy = accuracy_score(train_predictions, train_y)
    precision = precision_score(train_predictions, train_y)
    recall = recall_score(train_predictions, train_y)
    f1score = f1_score(train_predictions, train_y)
    train_predictions = np.where(train_predictions == 1, 'good review', 'bad review')

    print(f'Accuracy - {accuracy}\nPrecision - {precision}\nRecall - {recall}\nf1 score - {f1score}\n')

    train_fpr, train_tpr, _ = roc_curve(train_y, train_preds)
    test_fpr, test_tpr, _ = roc_curve(test_y, test_preds)

    train_auc, test_auc = np.round(auc(train_fpr, train_tpr), 4), np.round(auc(test_fpr, test_tpr), 4)

    plt.plot(train_fpr, train_tpr, label=f'AUC на обучающем наборе: {train_auc}')
    plt.plot(test_fpr, test_tpr, label=f'AUC на тестовом наборе: {test_auc}')
    plt.legend()
    plt.show()

    df = pd.DataFrame({'id': np.arange(len(train_predictions)), 'review': train_predictions})
    df.to_csv('output.csv', sep=',', header=0)
    with open('metrics.txt', 'w') as out:
        out.write(f'Accuracy - {accuracy}\nPrecision - {precision}\nRecall - {recall}\nf1 score - {f1score}\n')


LRegression()
