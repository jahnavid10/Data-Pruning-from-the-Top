# Data-Pruning-from-the-Top
'Data pruning' is common practice in recommender-systems research. We define data pruning as removing instances from a dataset that would not be removed in the real world, i.e. when used by recommender systems in production environments.

# Data Preparation
1. We have considered the MovieLens 100k Dataset and imported the necessary libraries like Pandas and Numpy.
2. Loaded the Dataset into Pandas Dataframe and created columns for the data in the appropriate data type.
3. Removed Top Users by taking 20% of the top users.
4. Now convert the filtered data from explicit to implicit feedback data by creating a new column for implicit feedback converting all the data into implicit ratings and saving it as a CSV file.
5. We get Pruned Implicit data which doesn't have the top user ratings.

# Evaluation using Sci-kit Library
1. I first used the sci-kit library to evaluate the Implicit data we have by loading and splitting the data into train and test sets and creating a user-item matrix 
   for the KNN algorithm and finding the k nearest neighbors by using cosine metrics(25 in this case).
2. Then used the Matrix Factorization method for the SVD Algorithm.
3. KNN and SVD Predictions are done using a for loop creating a new column for predictions in test data and providing a minimum and maximum scale for the 
   prediction ratings.
4. Evaluate both the models and print the RMSE, MAE, and NDCG values.
5. Also, the predicted ratings are also printed.

# Evaluation using Surprise Library
Surprise library is used for evaluating Recommender Systems.
1. Libraries are loaded.
2. The file path of the data is loaded from a CSV file using a Reader.
3. Data is split into train and test sets with an 80-20 ratio.
4. The SVD algorithm is used with an epoch of 50 and the data is trained using algo.fit and predictions are done.
5. Information of the predicted rating is printed and RMSE and MAE values are computed.
