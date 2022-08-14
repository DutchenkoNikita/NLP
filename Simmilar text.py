import re

import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



def preprocessing(data):
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    nltk.download('stopwords')

    def clear_text(text):
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
        review = ' '.join(review)
        return review

    data = data.apply(lambda x: clear_text(x))
    print(data)

    return data


def get_corpus():
    dt = pd.read_csv('people_wiki.csv', sep=',', header=0)
    inp = ['Barack Obama']
    request = pd.DataFrame({
        'URI': '-',
        'name': 'request',
        'text': inp})
    frames = [request, dt]
    dt = pd.concat(frames, ignore_index=True, sort=False)
    data = dt['text']
    print(data)
    data = preprocessing(data)
    tf_idf = TfidfVectorizer()
    vectors = tf_idf.fit_transform(data)
    vectors = tf_idf.transform(data)
    print(vectors)
    nn = NearestNeighbors(metric='cosine')
    nn.fit(vectors)
    request_index = dt[dt['name'] == 'request'].index[0]
    distances, indices = nn.kneighbors(vectors[request_index], n_neighbors=11)
    neighbors = pd.DataFrame({'distance': distances.flatten(), 'id': indices.flatten()})
    print(neighbors)
    nearest_info = (
        dt.merge(neighbors, right_on='id', left_index=True).sort_values('distance')[['id', 'text', 'distance']])
    print(nearest_info)
    with open('output.txt', 'w') as out:
        out.write(f'Request: {inp}\n')
        for i in range(len(nearest_info['text'])):
            out.write('#{}:{}\n'.format(i, nearest_info['text'][i]))


get_corpus()
