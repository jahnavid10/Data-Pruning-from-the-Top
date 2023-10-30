# Data-Pruning-from-the-Top
'Data pruning' is common practice in recommender-systems research. We define data pruning as removing instances from a dataset that would not be removed in the real world, i.e. when used by recommender systems in production environments.

# Data Preparation
1. We have considered the MovieLens 100k Dataset and imported the necessary libraries like Pandas and Numpy.
2. Loaded the Dataset into Pandas Dataframe and created columns for the data in the appropriate data type.
3. Removed Top Users by setting a threshold of the number of ratings to define the top users, and filter the data by excluding the top user ratings.
4. Now convert the filtered data from explicit to implicit feedback data by providing a threshold creating a new column for implicit feedback and saving it as a CSV file.
5. We get Pruned Implicit data which doesn't have the top user ratings.
