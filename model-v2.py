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
# library of kumparan
from kumparanian import ds
# sastrawi library, using for stopword removal
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# library to create tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
# library to create words count vectorize
from sklearn.feature_extraction.text import CountVectorizer
# library to split data
from sklearn.model_selection import train_test_split
# library to remove punctuation in every content
from nltk.tokenize import RegexpTokenizer
# accuracy metrics to evaluate prediction
from sklearn.metrics import accuracy_score
# library to learn model using Support Vector Machine
from sklearn.svm import SVC
# library to read data
import pandas as pd

class Model:

    def __init__(self): 
        """
        You can add more parameter here to initialize your model
        """
        # initialize the variable for count_vectorizer function
        self.count_vect = CountVectorizer()
        # initialize the variable for tf-idf function
        self.tfidf_transformer = TfidfTransformer()
        # initialize SVM classifier method
        self.svm_clf = SVC(kernel = 'linear', C = 1)

    def train(self):
        """
        NOTE: Implement your training procedure in this method.
        """
        # read data.csv using pandas and drop nan
        data = pd.read_csv("data.csv").dropna()
        # get article_content and transform to list
        contents = data["article_content"].values.tolist()
        # get article_topic and transform to list
        topics = data["article_topic"].values.tolist()
        # import library to tokenize and remove punctuation
        tokenizer = RegexpTokenizer(r'\w+')
        # stopword removal for bahasa indonesia
        stopword = StopWordRemoverFactory().create_stop_word_remover()
        # list to save clean contents
        clean_contents = list()
        # looping the contents, and preprocess for each content
        for content in contents:
            # case folding the sentence to be lowcase
            lowcase_word = content.lower() 
            # remove stopword from the content
            stop_word = stopword.remove(lowcase_word)      
            # tokenize the content
            sentence_token = tokenizer.tokenize(stop_word)
            # initialize a list for clean token 
            clean_tokens = list()       
            for token in sentence_token:
                # append token to the list after lower it
                clean_tokens.append(token)
            # transform a token to be sentence
            sentence = " ".join(clean_tokens) + ''
            # append clean sentence
            clean_contents.append(sentence)
        
        # count vectorizer
        X_train_counts = self.count_vect.fit_transform(clean_contents)
        # create tfidf from count vectorizer
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)      
        # split data to train and test set > test 10%, train 90%
        X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, topics, test_size=0.1)
        # train a model
        self.svm_clf.fit(X_train, y_train)

        # prediction for x_test
        prediction = self.svm_clf.predict(X_test)
        # model accuracy for x_test  
        accuracy = accuracy_score(y_test, prediction)
        # print accuracy
        print(accuracy)


    def predict(self, input):
        """
        NOTE: Implement your predict procedure in this method.
        """
        # put the input (string) into a list
        input_test = list(input)
        # count vectorize of input_test vocabulary
        new_counts = self.count_vect.transform(input_test)
        # create tf-idf for input test
        new_tfidf = self.tfidf_transformer.transform(new_counts)
        # get a prediction label
        label = self.svm_clf.predict(new_tfidf)[0]

        return label

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

    # try to predict
    article_topic = model.predict("article_content")
    print(article_topic)

    # Save your trained model to model.pickle
    model.save()