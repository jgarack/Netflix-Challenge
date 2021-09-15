import numpy as np
import pandas as pd

from utils import Utils


class CollaborativeFiltering:
    """
    Collaborative filtering Item-Item methods wrapper.
    """

    @staticmethod
    def predict_collaborative_filtering_item_item(movies, users, ratings, predictions, baseline=False):
        """
        Invoke collaborative filtering methods.

        :param baseline: Use of baseline toggle.
        :param movies: Movies data.
        :param users: users data.
        :param ratings: Ratings data.
        :param predictions: Predictions to make.
        :return: The predicted values.
        """
        utility_file = './data/utility.csv'
        similarity_movies_file = 'data/similarity_movies.csv'

        R = pd.read_csv(utility_file)

        baselines = None
        if baseline:
            baselines = Utils.generate_baselines(R, ratings, movies, users)

        return CollaborativeFiltering\
            .collaborative_filtering_item_item(R, similarity_movies_file, predictions, baselines)

    @staticmethod
    def collaborative_filtering_item_item(R, similarity_movies_file, predictions, baselines):
        """
        Item-Item collaborative filtering.

        :param baselines: The baselines to use.
        :param R: Identity matrix.
        :param similarity_movies_file: Similarity matrix file.
        :param predictions: Predictions to make.
        :return: The predicted values.
        """

        # Load similarity matrix.
        # S_movies = R.T.corr(method="pearson")
        # S_movies.to_csv(similarity_movies_file, index=False)
        S_movies = pd.read_csv(similarity_movies_file)
        # print(S_movies)

        # Get 100 nearest neighbors.
        S_movies_not_na = S_movies.fillna(0)
        i = S_movies_not_na.index.values
        v = S_movies_not_na.values
        n = 101
        K_nn = pd.DataFrame(i[v.argsort(0)[::-1]][:n], columns=S_movies_not_na.columns)

        # Predict rating.
        count = 1
        return_pred = []
        for k in range(len(predictions)):
            user_idx = predictions["userID"][k] - 1
            movie_idx = predictions["movieID"][k] - 1

            sim_rating_sum = 0
            sim_sum = 0

            # Prepare baselines, if needed.
            global_baseline = 0.0
            local_baseline = 0.0
            if baselines is not None:
                global_baseline = baselines["mean_rating"] \
                                  + baselines["mean_users"][user_idx] - baselines["mean_rating"] \
                                  + baselines["mean_movies"][movie_idx] - baselines["mean_rating"]

            for i in range(1, n):
                n_movie_idx = K_nn[str(movie_idx)][i]

                # Prepare baseline, if needed
                if baselines is not None:
                    local_baseline = baselines["mean_rating"] \
                                     + baselines["mean_users"][user_idx] - baselines["mean_rating"] \
                                     + baselines["mean_movies"][n_movie_idx] - baselines["mean_rating"]

                if R[str(user_idx)][n_movie_idx] > 0.0:
                    sim_rating_sum = sim_rating_sum \
                                     + S_movies_not_na[str(movie_idx)][n_movie_idx] \
                                     * (R[str(user_idx)][n_movie_idx] - local_baseline)
                    sim_sum = sim_sum + S_movies_not_na[str(movie_idx)][n_movie_idx]

            if sim_sum > 0:
                prediction = global_baseline + (sim_rating_sum / sim_sum)
            else:
                if baselines is not None:
                    prediction = baselines["mean_movies"][movie_idx]
                else:
                    prediction = 1.0

            # NaN prevention when using baselines.
            if baselines is not None:
                if np.isnan(prediction):
                    prediction = baselines["mean_users"][user_idx]

                if np.isnan(prediction):
                    prediction = baselines["mean_rating"]

            return_pred.append([count, prediction])
            count = count + 1

        return return_pred

    @staticmethod
    def collaborative_filtering_item_item_extended_baselines(R, similarity_movies_file, predictions, movies, users, baselines):
        """
        Item-Item collaborative filtering using extended baselines.

        :param baselines: The baselines to use.
        :param R: Identity matrix.
        :param similarity_movies_file: Similarity matrix file.
        :param predictions: Predictions to make.
        :return: The predicted values.
        """

        # Load similarity matrix.
        # S_movies = R.T.corr(method="pearson")
        # S_movies.to_csv(similarity_movies_file, index=False)
        S_movies = pd.read_csv(similarity_movies_file)
        # print(S_movies)

        # Get 100 nearest neighbors.
        S_movies_not_na = S_movies.fillna(0)
        i = S_movies_not_na.index.values
        v = S_movies_not_na.values
        n = 101
        K_nn = pd.DataFrame(i[v.argsort(0)[::-1]][:n], columns=S_movies_not_na.columns)

        # Predict rating.
        count = 1
        return_pred = []
        for k in range(len(predictions)):
            user_idx = predictions["userID"][k] - 1
            movie_idx = predictions["movieID"][k] - 1

            # Retrieve details for baselines.
            movie_year = str(movies["year"][movie_idx])
            user_gender = users["gender"][user_idx]
            user_age = str(users["age"][user_idx])
            user_profession = str(users["profession"][user_idx])

            sim_rating_sum = 0
            sim_sum = 0

            # Prepare baselines, if needed.
            global_baseline = 0.0
            local_baseline = 0.0
            if baselines is not None:
                global_baseline = baselines["mean_rating"] \
                                  + baselines["mean_users"][user_idx] - baselines["mean_rating"]\
                                  + baselines["mean_movies"][movie_idx] - baselines["mean_rating"] \
                                  + baselines["mean_years"][movie_year] - baselines["mean_rating"] \
                                  + baselines["mean_gender"][user_gender] - baselines["mean_rating"] \
                                  + baselines["mean_age"][user_age] - baselines["mean_rating"] \
                                  + baselines["mean_profession"][user_profession] - baselines["mean_rating"]

            for i in range(1, n):
                n_movie_idx = K_nn[str(movie_idx)][i]

                # Retrieve details for baselines.
                n_movie_year = str(movies["year"][n_movie_idx])

                # Prepare baseline, if needed
                if baselines is not None:
                    local_baseline = baselines["mean_rating"] \
                                     + baselines["mean_users"][user_idx] - baselines["mean_rating"] \
                                     + baselines["mean_movies"][n_movie_idx] - baselines["mean_rating"] \
                                     + baselines["mean_years"][n_movie_year] - baselines["mean_rating"] \
                                     + baselines["mean_gender"][user_gender] - baselines["mean_rating"] \
                                     + baselines["mean_age"][user_age] - baselines["mean_rating"] \
                                     + baselines["mean_profession"][user_profession] - baselines["mean_rating"]

                if R[str(user_idx)][n_movie_idx] > 0.0:
                    sim_rating_sum = sim_rating_sum \
                                     + S_movies_not_na[str(movie_idx)][n_movie_idx] \
                                     * (R[str(user_idx)][n_movie_idx] - local_baseline)
                    sim_sum = sim_sum + S_movies_not_na[str(movie_idx)][n_movie_idx]

            if sim_sum > 0:
                prediction = global_baseline + (sim_rating_sum / sim_sum)
            else:
                if baselines is not None:
                    prediction = baselines["mean_movies"][movie_idx]
                else:
                    prediction = 1.0

            # NaN prevention when using baselines.
            if baselines is not None:
                if np.isnan(prediction):
                    prediction = baselines["mean_users"][user_idx]

                if np.isnan(prediction):
                    prediction = baselines["mean_rating"]

            return_pred.append([count, prediction])
            count = count + 1

        return return_pred

    @staticmethod
    def collaborative_filtering_item_item_weights(R, similarity_movies_file, predictions, baselines):
        """
        Item-Item collaborative filtering using interpolated weights.

        :param baselines: The baselines to use.
        :param R: Identity matrix.
        :param similarity_movies_file: Similarity matrix file.
        :param predictions: Predictions to make.
        :return: The predicted values.
        """

        # Load weights matrix.
        weights_file = './data/weights.csv'
        W = pd.read_csv(weights_file)

        # Load similarity matrix.
        # S_movies = R.T.corr(method="pearson")
        # S_movies.to_csv(similarity_movies_file, index=False)
        S_movies = pd.read_csv(similarity_movies_file)
        # print(S_movies)

        # Get 100 nearest neighbors.
        S_movies_not_na = S_movies.fillna(0)
        i = S_movies_not_na.index.values
        v = S_movies_not_na.values
        n = 11
        K_nn = pd.DataFrame(i[v.argsort(0)[::-1]][:n], columns=S_movies_not_na.columns)

        # Predict rating.
        count = 1
        return_pred = []
        for k in range(len(predictions)):
            user_idx = predictions["userID"][k] - 1
            movie_idx = predictions["movieID"][k] - 1
            weighted_sum = 0

            # Prepare baselines, if needed.
            global_baseline = 0.0
            local_baseline = 0.0
            if baselines is not None:
                global_baseline = baselines["mean_rating"] \
                                  + baselines["mean_users"][user_idx] - baselines["mean_rating"] \
                                  + baselines["mean_movies"][movie_idx] - baselines["mean_rating"]

            for i in range(1, n):
                n_movie_idx = K_nn[str(movie_idx)][i]

                # Prepare baseline, if needed
                if baselines is not None:
                    local_baseline = baselines["mean_rating"] \
                                     + baselines["mean_users"][user_idx] - baselines["mean_rating"] \
                                     + baselines["mean_movies"][n_movie_idx] - baselines["mean_rating"]

                if R[str(user_idx)][n_movie_idx] > 0.0 and W[str(movie_idx)][n_movie_idx] != 1:
                    weighted_sum = weighted_sum \
                                     + W[str(movie_idx)][n_movie_idx] \
                                     * (R[str(user_idx)][n_movie_idx] - local_baseline)

            if weighted_sum > 0:
                prediction = global_baseline + weighted_sum
            else:
                if baselines is not None:
                    prediction = baselines["mean_movies"][movie_idx]
                else:
                    prediction = 1.0

            # NaN prevention when using baselines.
            if baselines is not None:
                if np.isnan(prediction):
                    prediction = baselines["mean_users"][user_idx]

                if np.isnan(prediction):
                    prediction = baselines["mean_rating"]

            return_pred.append([count, prediction])
            count = count + 1

        return return_pred
