import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def email_process():
    email = pd.read_csv('./0_Data_Generation/data/SpamCollection.csv', sep='\t', header=None, names=['Label', 'Content'])
    train, test = train_test_split(email, test_size = 0.25, random_state = 0)

    train['Content'] = train['Content'].str.replace('\W', ' ')
    train['Content'] = train['Content'].str.lower()
    train['Content'] = train['Content'].str.split()

    vocabulary = []
    for sms in train['Content']:
        for word in sms:
            vocabulary.append(word)
    vocabulary = list(set(vocabulary))

    return vocabulary, train, test

class Naive_bayes():
    def __init__(self) -> None:
        pass

    def fit(self, data, vocabulary):
        word_counts_per_sms = {unique_word: [0] * len(data['Content']) for unique_word in vocabulary}

        for index, sms in enumerate(data['Content']):
            for word in sms:
                word_counts_per_sms[word][index] += 1

        word_counts = pd.DataFrame(word_counts_per_sms)
        word_counts = word_counts.reset_index(drop=True) # !!
        data = data.reset_index(drop=True) # !!
        data_clean = pd.concat([data, word_counts], axis=1)

        # Isolating spam and ham messages first
        spam_messages = data_clean[data_clean['Label'] == 'spam']
        ham_messages = data_clean[data_clean['Label'] == 'ham']

        # P(Spam) and P(Ham)
        self.p_spam = len(spam_messages) / len(data_clean)
        self.p_ham = len(ham_messages) / len(data_clean)

        # N_Spam
        n_words_per_spam_message = spam_messages['Content'].apply(len)
        n_spam = n_words_per_spam_message.sum()

        # N_Ham
        n_words_per_ham_message = ham_messages['Content'].apply(len)
        n_ham = n_words_per_ham_message.sum()

        # N_Vocabulary
        n_vocabulary = len(vocabulary)

        # Laplace smoothing
        alpha = 1

        # Initiate parameters
        self.parameters_spam = {unique_word:0 for unique_word in vocabulary}
        self.parameters_ham = {unique_word:0 for unique_word in vocabulary}

        # Calculate parameters
        for word in vocabulary:
            n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
            p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
            self.parameters_spam[word] = p_word_given_spam

            n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
            p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
            self.parameters_ham[word] = p_word_given_ham

    def predict(self, message):
        message = re.sub('\W', ' ', message)
        message = message.lower().split()

        p_spam_given_message = self.p_spam
        p_ham_given_message = self.p_ham

        for word in message:
            if word in self.parameters_spam:
                p_spam_given_message *= self.parameters_spam[word]

            if word in self.parameters_ham: 
                p_ham_given_message *= self.parameters_ham[word]

        print('P(Spam|message):', p_spam_given_message)
        print('P(Ham|message):', p_ham_given_message)

        if p_ham_given_message >= p_spam_given_message:
            print('Label: Ham')
        elif p_ham_given_message < p_spam_given_message:
            print('Label: Spam')

        

if __name__ == "__main__":
    vocabulary, train, test = email_process()
    nb = Naive_bayes()
    nb.fit(train, vocabulary)
    nb.predict('WINNER!! This is the secret code to unlock the money: C3421.')