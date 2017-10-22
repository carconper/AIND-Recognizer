from my_recognizer import recognize
import time
import traceback
from asl_utils import show_errors
from my_model_selectors import SelectorDIC, SelectorBIC, SelectorCV
import numpy as np
import pandas as pd
from asl_data import AslDb

features = ['left-x', 'right-x', 'left-y', 'right-y']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
features_handdist = ['hand_dist']
features_nosedist = ['lnose_dist', 'rnose_dist']
features_custom = features_handdist + features_nosedist
selectors = [SelectorCV, SelectorDIC, SelectorBIC]
feature_sets = [features_norm, features_polar, features_delta, features_custom,
               features_handdist, features_nosedist]


def add_features1(asl):
	add_features_ground(asl)
	add_features_norm(asl)
	add_features_polar(asl)
	add_features_delta(asl)
	add_features_custom(asl)


def add_features_ground(asl):
    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

def add_features_norm(asl):
    df_means = asl.df.groupby('speaker').mean()
    df_std = asl.df.groupby('speaker').std()
    for speaker in asl.df.speaker.unique():
        for f in features:
            asl.df[f + '-mean']= asl.df['speaker'].map(df_means[f])
            asl.df[f + '-std']= asl.df['speaker'].map(df_std[f])
    asl.df[features_norm[0]] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
    asl.df[features_norm[1]] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
    asl.df[features_norm[2]] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
    asl.df[features_norm[3]] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']

def add_features_polar(asl):
    asl.df[features_polar[0]] = np.sqrt((asl.df['right-x'] - asl.df['nose-x']) **2  + (asl.df['right-y' ] - asl.df['nose-y']) ** 2)
    asl.df[features_polar[1]] = np.arctan2(asl.df['right-x'] - asl.df['nose-x'], asl.df['right-y'] - asl.df['nose-y'])
    asl.df[features_polar[2]] = np.sqrt((asl.df['left-x'] - asl.df['nose-x']) **2  + (asl.df['left-y'] - asl.df['nose-y']) ** 2)
    asl.df[features_polar[3]] = np.arctan2(asl.df['left-x'] - asl.df['nose-x'], asl.df['left-y'] - asl.df['nose-y'])

def add_features_delta(asl):
    asl.df['delta-rx'] = asl.df.groupby(level=[0])['right-x'].diff(1).fillna(0)
    asl.df['delta-ry'] = asl.df.groupby(level=[0])['right-y'].diff(1).fillna(0)
    asl.df['delta-lx'] = asl.df.groupby(level=[0])['left-x'].diff(1).fillna(0)
    asl.df['delta-ly'] = asl.df.groupby(level=[0])['left-y'].diff(1).fillna(0)

def add_features_custom(asl):
    asl.df['hand_dist'] = np.sqrt((asl.df['left-x'] - asl.df['right-x']) ** 2 + (asl.df['left-y'] - asl.df['right-y']) ** 2)
    asl.df['lnose_dist'] = np.sqrt((asl.df['left-x'] - asl.df['nose-x']) ** 2 + (asl.df['left-y'] - asl.df['nose-y']) ** 2)
    asl.df['rnose_dist'] = np.sqrt((asl.df['right-x'] - asl.df['nose-x']) ** 2 + (asl.df['right-y'] - asl.df['nose-y']) ** 2)


def train_all_words(features, model_selector):
    """
    Method used to generate a model dictionary containing models for all the
    words in our word set
    """
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, xlengths, word,
                               n_constant=3).select()
        model_dict[word] = model
    return model_dict




if __name__ == "__main__":

    asl = AslDb()
    add_features1(asl)
    for feats in feature_sets:
        print("===============================================")
        print(" FEATURES")
        print(" {}".format(feats))
        print("===============================================")
        for sel in selectors:
            print(" Selector: {}".format(sel))
            start_t = time.time()
            print("Start time: {}".format(start_t))
            print("--------------------------")
            try:
                models = train_all_words(feats, sel)
                test_set = asl.build_test(feats)
                probabilities, guesses = recognize(models, test_set)
                print(show_errors(guesses, test_set))
                end_t = time.time()
                total_t = end_t - start_t
                print("End time: {}".format(end_t))
                print("Total time in seconds: {}".format(total_t))
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
                print("There was some kind of problem with the specified selector")
            print("--------------------------")
