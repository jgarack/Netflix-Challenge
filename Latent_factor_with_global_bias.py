import numpy as np
import pandas as pd
from random import randint
from math import sqrt
from collaborative_filtering import CollaborativeFiltering

# -*- coding: utf-8 -*-
"""
FRAMEWORK FOR DATAMINING CLASS

#### IDENTIFICATION
NAME + SURNAME: Mattia Bonfanti, Jonathan Garack
STUDENT ID: 5002273, 4889045
KAGGLE ID: Group4


### NOTES
This files is an example of what your code should look like. 
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = 'data/submission.csv'
utility_file = './data/utility.csv'
similarity_users_file = './data/similarity_users.csv'
similarity_movies_file = 'data/Output/similarity_movies.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])

# so it is more time efficient we can save the utility matrix as csv an reuse it.
# R = create_user_movie_matrix(movies_description, users_description, ratings_description)
# R.to_csv(utility_file, index=False)

# read already crated csv file
R = pd.read_csv(utility_file)


#####
##
## SAVE RESULTS
##
#####

# helper function to compute avg movie and user ratings (output saved as .csv)
# the first position in each generated array should be omitted
def compute_avg_movie_and_user_ratings():
    avg_overall = avg_movie_rating()
    ratings = np.loadtxt("data/ratings.csv", delimiter=";")
    sum_movie = np.zeros((3707, 1))
    sum_users = np.zeros((6041, 1))
    cnt_movie = np.zeros((3707, 1))
    cnt_users = np.zeros((6041, 1))

    for i in range(len(ratings)):
        user = int(ratings[i, 0])
        movie = int(ratings[i, 1])
        score = ratings[i, 2]
        sum_users[user, 0] = sum_users[user, 0] + score
        cnt_users[user, 0] = cnt_users[user, 0] + 1

        sum_movie[movie, 0] = sum_movie[movie, 0] + score
        cnt_movie[movie, 0] = cnt_movie[movie, 0] + 1

    np.savetxt("data/user_sum.csv", sum_users, delimiter=";")
    np.savetxt("data/user_cnt.csv", cnt_users, delimiter=";")
    np.savetxt("data/movie_sum.csv", sum_movie, delimiter=";")
    np.savetxt("data/movie_cnt.csv", cnt_movie, delimiter=";")

    avg_users = sum_users / cnt_users
    avg_movie = sum_movie / cnt_movie
    avg_users = np.nan_to_num(avg_users, nan=avg_overall)
    avg_movie = np.nan_to_num(avg_movie, nan=avg_overall)
    np.savetxt("data/user_avg.csv", avg_users, delimiter=";")
    np.savetxt("data/movie_avg.csv", avg_movie, delimiter=";")


# helper function to compute overall mean
def avg_movie_rating():
    ratings = np.loadtxt("data/ratings.csv", delimiter=";")
    ratings_sum = 0
    for i in range(len(ratings)):
        ratings_sum = ratings_sum + ratings[i, 2]
    return ratings_sum / len(ratings)


# compute RMSE
def rmse(r, p_t, q, avg_overall, avg_movies, avg_users):
    predictions_lf = np.zeros((3706, 6040))
    for i in range(len(predictions_lf)):
        for j in range(len(predictions_lf[i])):
            # compute predicted value
            p_V = np.dot(q[i], p_t[:, j])
            # compute gradient
            predictions_lf[i, j] = (p_V - avg_overall + avg_movies[i + 1] + avg_users[j + 1])

    r = np.nan_to_num(r)
    error = 0
    cnt = 0

    for i in range(len(predictions_lf)):
        for j in range(len(predictions_lf[i])):
            if r[i][j] != 0:
                pred_error = np.abs(predictions_lf[i][j] - r[i][j])
                error = error + pred_error ** 2
                cnt += 1.0
    return sqrt(error / cnt)


# latent factor with combined baseline
def predict_latent_factors(predictions, k_factors):
    final_r = R.to_numpy()
    learning_rate = 0.005
    q_final = np.ones((3706, k_factors)) * 0.1
    p_t_final = np.ones((k_factors, 6040)) * 0.1
    k = 0
    avg_overall = avg_movie_rating()
    avg_movies = np.loadtxt("data/movie_avg.csv")
    avg_users = np.loadtxt("data/user_avg.csv")
    new_rmse = rmse(final_r, p_t_final, q_final, avg_overall, avg_movies, avg_users)

    while True:
        for i in range(len(final_r)):
            for j in range(len(final_r[i])):
                if final_r[i, j] != 0:
                    # compute predicted value
                    p_v = np.dot(q_final[i], p_t_final[:, j])
                    # compute gradient
                    gradient = final_r[i][j] - (p_v - avg_overall + avg_movies[i + 1] + avg_users[j + 1])
                    q_final[i] = q_final[i] + learning_rate * (2 * gradient * p_t_final[:, j] - 0.1 * q_final[i])
                    p_t_final[:, j] = p_t_final[:, j] + learning_rate * (
                            2 * gradient * q_final[i] - 0.1 * p_t_final[:, j])

        old_rmse = new_rmse
        new_rmse = rmse(final_r, p_t_final, q_final, avg_overall, avg_movies, avg_users)
        if new_rmse + 0.001 > old_rmse:
            break
        k += 1

    result = np.zeros((90019, 2))
    avg_overall = avg_movie_rating()
    avg_movies = np.loadtxt("data/movie_avg.csv")
    avg_users = np.loadtxt("data/user_avg.csv")
    predictions = predictions.to_numpy()
    for i in range(len(predictions)):
        curr = predictions[i]
        result[i, 0] = i + 1
        p = p_t_final[:, curr[0] - 1]
        result[i, 1] = np.dot(p, q_final[curr[1] - 1]) - avg_overall + avg_users[curr[0]] + avg_movies[curr[1]]

    return result


def predict_collaborative_filtering(movies, users, ratings, predictions, baseline=False):
    return CollaborativeFiltering \
        .predict_collaborative_filtering_item_item(movies, users, ratings, predictions, baseline)


# Predictions based on latent factor
predictions_lf = predict_latent_factors(predictions_description, 50)

print("Latent Factor completed")

# Predictions based on collaborative filtering
predictions_cf = predict_collaborative_filtering(movies_description, users_description, ratings_description,
                                                 predictions_description, True)

print("Collaborative filtering completed")
# Combine the results
predictions_cf = np.asarray(predictions_cf)
predictions = predictions_cf
for i in range(len(predictions_cf)):
       predictions[i, 1] = 0.8 * predictions_cf[i, 1] + 0.2 * predictions_lf[i, 1]

predictions.tolist()
# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formats data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it down
    submission_writer.write(predictions)
