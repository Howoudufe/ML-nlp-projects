import numpy as np
from gensim.models import Word2Vec
from features import get_features_w2v
from classifier import search_C
from scipy.stats import randint, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from itertools import product


def search_hyperparams(Xt_train, y_train, Xt_val, y_val):
    """Search the best values of hyper-parameters for Word2Vec as well as the
    regularisation parameter C for logistic regression, using the validation set.

    Args:
        Xt_train, Xt_val (list(list(list(str)))): Lists of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens) for training and validation, respectively.
        y_train, y_val: Dense vectors (np.ndarray) of class labels for training
            and validation, respectively. Each element of the vector is either
            0 or 1.

    Returns:
        dict(str : union(int, float)): The best values of hyper-parameters.
    """
    # TODO: tune at least two of the many hyper-parameters of Word2Vec 
    #       (e.g. vector_size, window, negative, alpha, epochs, etc.) as well as
    #       the regularisation parameter C for logistic regression
    #       using the validation set.

    # The code below needs to be modified.
    # best_params = dict()
    # best_C = 1.  # sklearn default
    # best_acc = 0.

    # best_params['C'] = best_C

    # assert 'C' in best_params
    # return best_params
    # ==============================================

    # best_params = {}
    # best_acc = 0

    # # The parameter space
    # param_dist = {
    #     'vector_size': randint(50, 200),
    #     'window': randint(3, 10),
    #     'C': loguniform(50, 100) # According to former C search
    # }

    # # Define a function to create and evaluate a model with given parameters
    # def create_and_evaluate_model(vector_size, window, C):
    #     word_vectors = train_w2v(Xt_train, vector_size=vector_size, window=window)
    #     X_train = get_features_w2v(Xt_train, word_vectors)
    #     X_val = get_features_w2v(Xt_val, word_vectors)
        
    #     model = LogisticRegression(C=C, max_iter=1000)
    #     model.fit(X_train, y_train)
        
    #     return model.score(X_val, y_val)

    # # Perform randomized search
    # random_search = RandomizedSearchCV(
    #     estimator=LogisticRegression(),
    #     param_distributions=param_dist,
    #     n_iter=20,
    #     scoring=create_and_evaluate_model,
    #     n_jobs=-1,
    #     cv=3,
    #     random_state=42
    # )

    # # Fit the random search object
    # random_search.fit(Xt_train, y_train)

    # # Get the best parameters
    # best_params = random_search.best_params_
    
    # return best_params

    # Grid (w2v) + random (C)=========================
    # Define some potential parameters
    vector_sizes = [100, 200, 300]
    windows = [3, 5, 7]
    
    best_params = {}
    best_acc = 0
    
    # Use grid search
    for vector_size, window in product(vector_sizes, windows):
        print(f"Training Word2Vec with vector_size={vector_size}, window={window}")
        
        # Train the Word2Vec model
        sentences_train = [sent for doc in Xt_train for sent in doc]
        w2v_model = Word2Vec(sentences=sentences_train, vector_size=vector_size, window=window, min_count=5, workers=4)
        
        # Generate features for train and validation sets
        X_train = get_features_w2v(Xt_train, w2v_model.wv)
        X_val = get_features_w2v(Xt_val, w2v_model.wv)
        
        # Given current Word2Vec features, search for the best C value 
        best_C, acc = search_C(X_train, y_train, X_val, y_val, return_best_acc=True)
        
        print(f"Best C: {best_C}, Accuracy: {acc}")
        
        if acc > best_acc:
            best_acc = acc
            best_params = {
                'vector_size': vector_size,
                'window': window,
                'C': best_C
            }
    
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_acc}")
    
    return best_params


def train_w2v(Xt_train, vector_size=200, window=5, min_count=5, negative=10, epochs=3, seed=101, workers=10,
              compute_loss=False, **kwargs):
    """Train a Word2Vec model.

    Args:
        Xt_train (list(list(list(str)))): A list of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens).
        See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
        for descriptions of the other arguments.

    Returns:
        gensim.models.keyedvectors.KeyedVectors: A mapping from words (string) to their embeddings
            (np.ndarray)
    """
    sentences_train = [sent for doc in Xt_train for sent in doc]

    # TODO: train the Word2Vec model
    print(f'Training word2vec using {len(sentences_train):,d} sentences ...')
 
    # The code below needs to be modified.
    # w2v_model = None

    w2v_model = Word2Vec(sentences=sentences_train,
                         vector_size=vector_size,
                         window=window,
                         min_count=min_count,
                         negative=negative,
                         epochs=epochs,
                         seed=seed,
                         workers=workers,
                         compute_loss=compute_loss,
                         **kwargs)

    return w2v_model.wv

