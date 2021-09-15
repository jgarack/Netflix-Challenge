# Netflix-Challenge  
*Netflix Challenge for TU Delft CSE2525 Data Mining.*  
*Mattia Bonfanti (5002273)* 
*Jonathan Garack (4889045)*  

The Netflix Prize was a competition for the best algorithm to predict user ratings for films
using only previous user ratings. The competition was held between 2006 and 2009 with the final
prize of $1,000,000 awarded to the algorithm with the best improvement over Netflix’s own
“Cinematch” algorithm. The winning algorithm presented a Test RMSE of 0.8567 with an
improvement of over 10% compared to “Cinematch”. [1]

***
##Code structure
the assignment consists of 4 .py files:

* **Latent_factor_with_global_bias.py** - implementation of Latent factor model with global effect and user/movie biases
* **Collaborative_filtering.py** - implementation of collaborative filtering model + baselines
* **utils.py** - function to generate baselines for collaborative filtering

    
***
####main methods:
* *compute_avg_movie_and_user_ratings()* - computes the biases of the users and movies
* *avg_movie_rating()* - computes the global average
* *rmse()* - computes the RMSE of the given ratings compared to the true ratings
* *predict_latent_factors()* - factorizes the utility matrix into two thin matrices and performs SGD
* *predict_collaborative_filtering* - method to predict ratings base on collaborative filtering model
  * the code of the method is in *collaborative_filtering.py*
  
* *utils.py/generate_baselines()* - generates baselines for collaborative filtering (output saved in a *.csv* file)
* *utils.py/interpolated_weights()* - simplified gradient descent to compute the weights matrix (output saved in a *.csv* file)
