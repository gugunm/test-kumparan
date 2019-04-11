"""
Kumparan's Model Interface

This is an interface file to implement your model.

You must implement `train` method and `predict` method.

`train` is a method to train your model. You can read
training data, preprocess and perform the training inside
this method.

`predict` is a method to run the prediction using your
trained model. This method depends on the task that you
are solving, please read the instruction that sent by
the Kumparan team for what is the input and the output
of the method.

In this interface, we implement `save` method to helps you
save your trained model. You may not edit this directly.

You can add more initialization parameter and define
new methods to the Model class.

Usage:
Install `kumparanian` first:

    pip install kumparanian

Run

    python model.py

It will run the training and save your trained model to
file `model.pickle`.
"""

from kumparanian import ds
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import sklearn
import dill


class Model:

    def __init__(self): 
        """
        You can add more parameter here to initialize your model
        """
        # self.model = 
        pass

    def train(self):
        """
        NOTE: Implement your training procedure in this method.
        """
        # read data.csv using pandas
        data = pd.read_csv("data.csv").dropna()
        # get article_content and transform to list
        contents = data["article_content"].values.tolist()
        # get article_topic and transform to list
        topics = data["article_topic"].values.tolist()
        # import library untuk tokenisasi dan remove punctuation
        tokenizer = RegexpTokenizer(r'\w+')
        '''
        # stemmer untuk bahasa
        stemmer = StemmerFactory().create_stemmer()
        # stopword removal untuk bahasa
        stopword = StopWordRemoverFactory().create_stop_word_remover()
        '''
        # list to save clean contents
        clean_contents = list()
        # looping the contents, and preprocess for each content
        for content in contents:
            # case folding the sentence to be lowcase
            lowcase_word = content.lower() 
            '''
            # hapus stopword dari content
            stop_word = stopword.remove(content)      
            # stemm kata di content
            stemming  = stemmer.stem(stop_word)
            '''
            # tokenisasi content
            sentence_token = tokenizer.tokenize(lowcase_word)
            # inisiasi list utuk token yang sudah bersih
            clean_tokens = list()       
            for token in sentence_token:
                # append token ke dalma list after lower it
                clean_tokens.append(token)
            # mengubah token menjadi sentence lagi
            sentence = " ".join(clean_tokens) + ''
            # append clean content after tokenization
            clean_contents.append(sentence)
        
        print(len(contents), len(topics))
        # count vectorizer
        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(clean_contents)

        # create tfidf from count vectorizer
        self.tfidf_transformer = TfidfTransformer()
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
        
        # classification_method = MultinomialNB().fit(X_train_tfidf, training_data.flag)
        X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, topics, test_size=0) #, random_state=42)
        self.clf = MultinomialNB().fit(X_train, y_train)


    def predict(self, input):
        """
        NOTE: Implement your predict procedure in this method.
        """
        # ============================
        # docs_new = "Status calon haji"
        docs_new = [input]

        X_new_counts = self.count_vect.transform(docs_new)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        label = self.clf.predict(X_new_tfidf)
        # ============================

        # Examples; psuedocode
        # processed_input = process_input(input)
        # output = self.network.forward(processed_input)
        # label = get_label(output)
        return label[0]

    def save(self):
        """
        Save trained model to model.pickle file.
        """
        ds.model.save(self, "model.pickle")


if __name__ == '__main__':
    # NOTE: Edit this if you add more initialization parameter
    model = Model()

    # Train your model
    model.train()

    # Save your trained model to model.pickle
    model.save()

    # try to predict
    print(model.predict("status calon haji kami gagal"))
