import pandas as pd
import numpy as np
import graphlab as gl
from bs4 import BeautifulSoup

def _get_len_joke(filename):
    with open(filename, 'rt') as f:
        jokes=BeautifulSoup(f)
        p = jokes.body.find_all('p')[1:]
        p = map(lambda x:x.text,p)
        len_jokes = map(lambda x:len(x),p)
        return  gl.SFrame({'joke_id': range(1,151),
                                 'len_jokes': len_jokes})

def data_pruning(df, cutoff=8):
    filtered_df = df.groupby('user_id').count()
    dense_matrix = filtered_df[filtered_df['rating'] > cutoff]
    user_ids = dense_matrix.index
    ratings = gl.SFrame(df[df['user_id'].isin(np.asarray(user_ids))])
    return ratings

def grid_searcher(train, test, params, model=gl.ranking_factorization_recommender.create, max_models=5):
    job = gl.model_parameter_search.create((train, test), model, params, max_models=max_models)
    return job.get_results()


# def glorified_gridSearch(data):
#    params = {'user_id':'user_id', 'item_id':'joke_id' 'target': 'rating', 'num_factors':[20,22,24,26,28,30,32,34,35,37,39],'max_num_users':[3000,4000,5000,6000,7000],'max_iterations':[20,25,30],
#    'solver':['ials','adagrad','sgd']}
#    job = graphlab.toolkits.model_parameter_search.create(
#        (training, validation), ranking_factorization_recommender.create, params)
#    results = job.get_results()
#    return results.column_names()


if __name__ == "__main__":
    ratings_data_fname = "data/ratings.dat"
    validation_data_fname = "data/validation_data.csv"

    # df = pd.read_table(ratings_data_fname)
    # ratings = data_pruning(df)

    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    validation = gl.SFrame(validation_data_fname, format='csv')

    # jokes = _get_len_joke("data/jokes.dat")

    params = {'target': 'rating',
          'user_id': 'user_id',
          'item_id': 'joke_id',
          'solver': 'auto',
          'regularization': list(np.logspace(-10,-5,5))}

    train, test = gl.recommender.util.random_split_by_user(ratings,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     max_num_users=5000,
                                                     item_test_proportion=0.05)

    # grid_searcher(train, test, params)



    rec_engine1 = gl.ranking_factorization_recommender.create(observation_data=ratings,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     solver='auto')

    rec_engine2 = gl.ranking_factorization_recommender.create(observation_data=ratings,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     solver='auto',
                                                     max_iterations= 25,
                                                     num_factors= 32,
                                                     num_sampled_negative_examples= 4,
                                                     ranking_regularization= 0.1,
                                                     regularization= 0.0001)


    gl.recommender.util.compare_models(validation, [rec_engine1, rec_engine2], model_names=['rec_enginge1', 'rec_engine2'])
