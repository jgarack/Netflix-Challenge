import numpy as np
import pandas as pd


class Utils:
    """
    Useful functions to use in the scripts.
    """

    @staticmethod
    def generate_baselines(utility_matrix, ratings, movies, users):
        """
        Generate baselines for collaborative filtering.

        :param utility_matrix: The utility matrix movies-users.
        :param ratings: The users-movies ratings.
        :return: The baselines object.
        """
        baselines = {}

        # Calculate the overall ratings mean.
        baselines["mean_rating"] = ratings["rating"].mean()

        # Replace zeros with Nan to exclude non rated movies from the mean calculation.
        df = utility_matrix.replace(0, np.NaN)

        # Calculate users mean ratings.
        baselines["mean_users"] = []
        for user in df:
            baselines["mean_users"].append(df[user].mean())

        # Calculate movies mean ratings.
        baselines["mean_movies"] = []
        for movie in df.T:
            baselines["mean_movies"].append(df.T[movie].mean())

        # Specific baselines.
        movie_years = {}
        user_genders = {}
        user_ages = {}
        user_professions = {}
        for i in range(len(ratings)):
            user_idx = ratings['userID'][i] - 1
            movie_idx = ratings['movieID'][i] - 1

            # Skip Nan.
            if np.isnan(df[str(user_idx)][movie_idx]):
                continue

            # Add movie years rating.
            year = movies["year"][movie_idx]
            if not np.isnan(year):
                if str(year) not in movie_years:
                    movie_years[str(year)] = []
                movie_years[str(year)].append(df[str(user_idx)][movie_idx])

            # Add gender ratings.
            gender = users["gender"][user_idx]
            if gender is not None:
                if gender not in user_genders:
                    user_genders[gender] = []
                user_genders[gender].append(df[str(user_idx)][movie_idx])

            # Add age ratings.
            age = users["age"][user_idx]
            if not np.isnan(age):
                if str(age) not in user_ages:
                    user_ages[str(age)] = []
                user_ages[str(age)].append(df[str(user_idx)][movie_idx])

            # Add age ratings.
            profession = users["profession"][user_idx]
            if not np.isnan(profession):
                if str(profession) not in user_professions:
                    user_professions[str(profession)] = []
                user_professions[str(profession)].append(df[str(user_idx)][movie_idx])

        # Calculate specific baselines.
        baselines["mean_years"] = {}
        baselines["mean_gender"] = {}
        baselines["mean_age"] = {}
        baselines["mean_profession"] = {}

        for k, v in movie_years.items():
            baselines["mean_years"][k] = np.array(v).mean()

        for k, v in user_genders.items():
            baselines["mean_gender"][k] = np.array(v).mean()

        for k, v in user_ages.items():
            baselines["mean_age"][k] = np.array(v).mean()

        for k, v in user_professions.items():
            baselines["mean_profession"][k] = np.array(v).mean()

        return baselines

    @staticmethod
    def interpolated_weights(R, similarity_movies_file, predictions, baselines):
        """
        Calculate the weighted interpolation between movies using gradient descent.

        :param baselines: The baselines to use.
        :param R: Identity matrix.
        :param similarity_movies_file: Similarity matrix file.
        :param predictions: Predictions to make.
        :return: The predicted values.
        """

        weights_file = './data/weights.csv'
        best_predictions_file = \
            './data/submissions/submission_final_0.862.csv'

        # Load best predictions.
        best_predictions = pd.read_csv(best_predictions_file)

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

        # Initialize weight matrix.
        movies_size = len(R.T.columns)
        W = pd.DataFrame(np.ones((movies_size, movies_size)))

        for pred_idx in range(len(predictions)):
            print(pred_idx)
            r_x_i = best_predictions["Rating"][pred_idx]
            user_x = predictions["userID"][pred_idx] - 1
            movie_i = predictions["movieID"][pred_idx] - 1
            for descent in range(13):
                for j in range(1, n):
                    movie_j = K_nn[str(movie_i)][j]
                    tot_weighted_sum = 0.0
                    # If user rated the movie, calculate the weight interpolation.
                    if R[str(user_x)][movie_j] > 0.0:
                        global_baseline = baselines["mean_rating"] \
                                          + baselines["mean_users"][user_x] - baselines["mean_rating"] \
                                          + baselines["mean_movies"][movie_i] - baselines["mean_rating"]

                        local_baseline_j = baselines["mean_rating"] \
                                         + baselines["mean_users"][user_x] - baselines["mean_rating"] \
                                         + baselines["mean_movies"][movie_j] - baselines["mean_rating"]

                        weighted_sum = 0.0
                        for k in range(1, n):
                            movie_k = K_nn[str(movie_i)][k]
                            if movie_k != movie_j and movie_k != movie_i:
                                local_baseline_k = baselines["mean_rating"] \
                                                   + baselines["mean_users"][user_x] - baselines["mean_rating"] \
                                                   + baselines["mean_movies"][movie_k] - baselines["mean_rating"]
                                if R[str(user_x)][movie_k] > 0.0:
                                    weighted_sum = weighted_sum + W[movie_i][movie_k] \
                                                   * (R[str(user_x)][movie_k]-local_baseline_k)

                        if weighted_sum > 0.0:
                            weighted_sum = weighted_sum + global_baseline
                            weighted_sum = weighted_sum - r_x_i
                            weighted_sum = weighted_sum * (R[str(user_x)][movie_j] - local_baseline_j)

                        tot_weighted_sum = tot_weighted_sum + weighted_sum

                    tot_weighted_sum = tot_weighted_sum * 2.0
                    w_old = W[movie_i][movie_j]
                    w_new = w_old - 0.0000001 * tot_weighted_sum
                    if np.abs(w_new - w_old) > 0.0000000001:
                        W[movie_i][movie_j] = w_new
                        W[movie_j][movie_i] = w_new

        print(W)
        W.to_csv(weights_file, index=False)
