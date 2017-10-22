"""This module implements the recognizer"""
import warnings
import numpy as np
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # DONE implement the recognizer
    #print(test_set.sentences_index)
    #print(test_set.wordlist, len(test_set.wordlist))
    #print("THE MODELS: ", models)
    #print(test_set._data)
    #print(test_set.num_items)
    for i, word in enumerate(test_set.wordlist):
        max_score = -np.inf
        probabilities.append({})
        for model in models:
            x_aux, lengths_aux = test_set.get_item_Xlengths(i)
            #print("WORD: ", model, x_aux, lengths_aux)
            try:
                score = models[model].score(x_aux, lengths_aux)
                probabilities[i][model] = score
            except:
                pass
            if score > max_score:
                result = (model, models[model], score)
                max_score = score
        guesses.append(result[0])
        #probabilities.append(result[2])
    #print("Probabilities dict: ", probabilities)
    #print("Test Set: ", test_set.wordlist)
    #print("Guesses List: ", guesses)
    return probabilities, guesses
