import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def cv_model(self, num_states, training_X, training_lengths):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #training_X, training_lengths = combine_sequences(training_fold_idx,
        #                                                 self.X)
        try:
            hmm_model = GaussianHMM(n_components=num_states,
                                    covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(training_X,
                                                       training_lengths)
            if self.verbose:
                print("training model created for {} with {} states based on\
                      dataset {}".format(self.this_word, num_states,
                                                 training_X))
            return hmm_model
        except:
            if self.verbose:
                print("model creation failed for {} with {} states based on\
                      dataset {}".format(self.this_word, num_states,
                                                 training_X))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        min_bic = np.inf
        winner = None
        word_seq = self.hwords[self.this_word]
        num_feat = len(word_seq[0][0])
        likelihood = 0
        for num_comp in range(self.min_n_components, self.max_n_components + 1):
            bmodel = self.base_model(num_comp)
            try:
                likelihood = bmodel.score(self.X, self.lengths)
                #likelihood = bmodel.score(word_seq[0], word_seq[1])
            except:
                pass
            # The number of free parameters is calculated based on the
            # following formula in our case:
            #   p = n*(n - 1) + (n - 1) + n*f + n*f
            #     = n**2 + 2*n*f
            #       where n is number of states
            #       and f is number of features
            #   SOURCE:
            #    - https://discussions.udacity.com/t/understanding-better-model-selection/232987/4
            #    - Slack comment form Dana Sheahen
            #     (...)There is one thing a little different for our project though...
            #     in the paper, the initial distribution is estimated and
            #     therefore those parameters are not "free parameters".
            #     However, hmmlearn will "learn" these for us if not provided.
            #     Therefore they are also free parameters:
            #       => p = n*(n-1) + (n-1) + 2*d*n
            #       = n^2 + 2*d*n - 1  (...)
            parameters = num_comp**2 + 2 * (num_comp * num_feat) - 1
            # N will be the number of data_points of the training set
            #data_points = len(word_seq[0]) * len(word_seq[0][0])
            data_points = len(self.X) * len(self.X[0])
            #print("Likelihood: {}".format(likelihood))
            #print("N and p: {} and {}". format(data_points, parameters))
            # Based on the slides provided:
            bic = -2 * likelihood + parameters * math.log(data_points)
            # Based on "Alain Biem"'s paper (with alpha=2):
            #bic = likelihood - parameters * math.log(data_points) / 2
            if bic < min_bic:
                min_bic = bic
                winner = bmodel
        return winner

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #min_dic = np.inf
        max_dic = -np.inf
        winner = None
        word_seq = self.hwords[self.this_word]
        num_feat = len(word_seq[0][0])
        rest_words = [word for word in self.words.keys() if word !=
                      self.this_word]
        #print("Words: {}".format(self.words))
        likelihood = 0
        for num_comp in range(self.min_n_components, self.max_n_components + 1):
            bmodel = self.base_model(num_comp)
            try:
                #likelihood = bmodel.score(word_seq[0])
                likelihood = bmodel.score(self.X, self.lengths)
            except:
                pass

            likelihoods = []
            for word in rest_words:
                try:
                    #lh = bmodel.score(self.hwords[word][0])
                    lh = bmodel.score(self.hwords[word][0], self.hwords[word][1])
                    likelihoods.append(lh)
                except:
                    pass
            dic = likelihood - sum([lh for lh in likelihoods]) / (len(likelihoods) - 1)
            if dic > max_dic:
                max_dic = dic
                winner = bmodel
        return winner


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        max_cv_lh = -np.inf
        winner = None
        likelihood = -np.inf
        if len(self.sequences) == 1:
            for num_comp in range(self.min_n_components, self.max_n_components + 1):
                bmodel = self.base_model(num_comp)
                try:
                    likelihood = bmodel.score(self.X, self.lengths)
                except:
                    pass
                if likelihood > max_cv_lh:
                    max_cv_lh = likelihood
                    winner = bmodel
            return winner
        elif len(self.sequences) == 2:
            split_method = KFold(2)
        else:
            split_method = KFold()
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            #print(cv_train_idx, self.X)
            x_train, lengths_train = combine_sequences(cv_train_idx,
                                                       self.sequences)
            x_test, lengths_test = combine_sequences(cv_test_idx,
                                                     self.sequences)
            #print("##### sequences ", self.sequences)
            #print("##### x_test, lengths_text ", x_test, lengths_test)
            for num_comp in range(self.min_n_components, self.max_n_components + 1):
                cvmodel = self.cv_model(num_comp, x_train, lengths_train)
                try:
                    likelihood = cvmodel.score(x_test, lengths_test)
                except:
                    pass
                if likelihood > max_cv_lh:
                    max_cv_lh = likelihood
                    winner = cvmodel
        return winner
