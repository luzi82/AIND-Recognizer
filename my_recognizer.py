import warnings
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
    # TODO implement the recognizer
    # return probabilities, guesses
    
    for id in range(test_set.num_items):
        X, lengths = test_set.get_item_Xlengths(id)
        
        prob = {}
        gues = None
        gues_score = float('-inf')
        
        for word in models:
            try:
                model = models[word]
                tmp_score = model.score(X,lengths)
                prob[word] = tmp_score
                if tmp_score > gues_score:
                    gues_score = tmp_score
                    gues = word
            except Exception as ex:
                # print('OURQILZE word={} ex={}'.format(word,ex))
                prob[word] = None
        
        probabilities.append(prob)
        guesses.append(gues)
    
    return probabilities, guesses
