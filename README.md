# Data Pruning from the Top- Do we need Top users for Good recommendations?
'Data pruning' is common practice in recommender-systems research. We define data pruning as removing instances from a dataset that would not be removed in the real world, i.e. when used by recommender systems in production environments.
The research that follows seeks to investigate how various data pruning thresholds impact the overall effectiveness of recommendation systems. The study examines the impact of omitting top users, who have a high number of reviews, on the performance of these systems in comparison to standard methods that employ unpruned datasets. We aim to analyze the performance of Implicit ALS and KNN algorithms in ranking metrics across multiple datasets with varied threshold percentages, Top-N values, and combinations of hyperparameters.

## Dependencies:
1. Python - 3.8.18
2. pandas - 1.5.3
3. numpy - 1.24.3
4. Lenskit - 0.14.2
5. Implicit - 0.7.2
6. Scikit-learn -1.3.2 
7. matplotlib - 3.7.4
8. Scipy - 1.10.1
9. Seaborn - 0.13.1

## Datasets:
1. Movielens-100k
2. Movielens latest small
3. Movielens 1M
   
## Algorithms:
1. Implicit ALS
2. KNN

## Parameters:
1. N = 3, 5, 10, 20
2. Combination of (factors, n_nbrs) = (25,10), (50,50), (75,25), (100,75)
3. Pruning threshold = 2%, 5%, 10%

## Procedure:
1. Importing Libraries - Pandas, numpy, Lenskit, Scikit-learn, Scipy, Implicit, Seaborn, and Matplotlib libraries are imported
2. Loading and preprocessing of data - The explicit data is converted into Implicit data and further it is pruned to a certain threshold.
3. Splitting the data - The data is split using the scikit-learn train test split function randomly by 80-20 and ensuring that at least one user is present in train data to avoid cold start problems.
4. User-item matrix - The train data and test data are converted into csr matrix for further usage in the Implicit ALS algorithm where rows are represented by users and columns are the items.
5. Training the models- The training matrix and train data are trained and fit over Implicit ALS and KNN algorithms respectively.
6. Generating User-item scores - The model.recommend function is used to generate recommendations in test data and accordingly scores and ranks are generated for each user based on Top-N value.
7. NDCG calculation - The individual NDCG values for each user and Mean NDCG values are calculated for both models.
8. NDCG plot - The mean NDCG is plotted to compare the results of both models.

This procedure is followed for: 
1. The original Implicit data on different N and (factors, n_nbrs) combinations.
2. The Pruned Implicit data on different Ptuning thresholds, N and (factors, n_nbrs) combinations.

## Results and Discussions:
1. Implicit ALS has a greater Mean NDCG than KNN due to its matrix factorization technique, allowing it to predict user preferences even for items they haven't interacted with.
2. KNN relies on neighborhood relationships, which might not capture the underlying structure of the data as effectively, especially when there are many items and complex user-item interactions.
3. Implicit ALS has a 5.145% lower Pruned Average Mean NDCG than Unpruned Average Mean NDCG over all datasets.
4. Implicit ALS could suffer more from data pruning compared to KNN due to its sensitivity to removing the most frequent users.
5. For different values of Top-N, factors, and nearest neighbors, Mean NDCG is higher than the rest of the combinations of Top-N and hyperparameters.
6. The Aggregate of all Mean NDCGs of Unpruned data is greater than Pruned data for all datasets at all Top-N, hyperparameters, and threshold percentages respectively.
7. For the KNN algorithm, the percentages are 1.931%, and 3.5007% for Movielens-100k and Movielens-1M, and 5.859% for Movielens-latest-small.
8. When the Data is 10% pruned from the Top, the percentage of Aggregate Unpruned data is 10.67%, 9.059%, and 7.982% higher than 10% Pruned data for Movielens 100k, Movielens-latest-small, and Movielens-1M Datasets respectively.
9. KNN suffers less than Implicit ALS but Movielens-1M suffers more than other datasets when the KNN algorithm is used.

## Conclusion:
The study compares Implicit ALS and KNN algorithms in recommender systems, focusing on Top-user data pruning and Unpruned data. It reveals that removing top users has a larger negative impact on Implicit ALS than KNN. The performance of Implicit ALS is influenced by the user-item interaction matrix structure, which is significantly altered by pruning. While pruning generally leads to a decrease in Mean NDCG for both algorithms, Implicit ALS experiences a more pronounced decrease. The analysis emphasizes the need for careful consideration of pruning techniques, as their effectiveness and impact can vary depending on the underlying algorithm and dataset characteristics.

### References: 
1.  Michael D Ekstrand. Lenskit for python: Next-generation software for recommender systems experiments. In Proceedings of the 29th ACM international conference on information & knowledge management, pages:2999-3006, 2020.




