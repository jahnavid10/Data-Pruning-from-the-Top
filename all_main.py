# Importing Libraries
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from lenskit.algorithms import Recommender
from lenskit.algorithms import item_knn
from lenskit.algorithms.item_knn import ItemItem
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import itertools
from all import *


# Global variables
N_values = [3, 5, 10, 20]
hyperparameters = [(25,10), (50,50), (75,25), (100,75)]
threshold_percentages = [0.98, 0.95, 0.90]

# Initialize a list to store results
implicit_results = []
pruned_implicit_results = []

# Main function 
def main():
    # Loading and preprocessing the data
    file_path = r"/home/g057635/arbeit/datasets/MovieLens-1M.csv"
    implicit_data = load_and_convert_to_implicit(file_path)
    print("Implicit Data:\n", implicit_data.head())

    for N, (factors, n_nbrs) in itertools.product(N_values, hyperparameters):
        # Splitting the data and determining the complete set of unique users and items
        all_implicit_users = implicit_data['user'].unique()
        all_implicit_items = implicit_data['item'].unique()
        implicit_train_data, implicit_test_data = train_test_split_data(implicit_data, train_size=0.8, random_state=42)

        # User-item interaction matrix
        implicit_train_matrix = create_user_item_matrix(implicit_train_data, all_implicit_users, all_implicit_items)

        # Training the Model
        implicit_als_model = train_als_model(implicit_train_matrix, factors)
        implicit_knn_model = train_knn_model(implicit_train_data, n_nbrs)

        # Generating recommendations for both the models
        implicit_test_user_ids = implicit_test_data['user'].unique()
        implicit_als_recommendations = generate_als_user_item_scores(implicit_als_model, implicit_train_matrix, implicit_test_user_ids, N, filter_already_liked_items=True)
        implicit_knn_recommendations = generate_knn_recommendations(implicit_knn_model, implicit_train_data, implicit_test_data, N)

        # Individual NDCG scores
        implicit_individual_als_ndcg_scores = calculate_individual_ndcg(implicit_als_recommendations, implicit_test_data, N)  
        implicit_individual_knn_ndcg_scores = calculate_individual_ndcg(implicit_knn_recommendations, implicit_test_data, N)

        # Mean NDCG score
        implicit_mean_als_ndcg = np.mean(list(implicit_individual_als_ndcg_scores.values()))
        implicit_mean_knn_ndcg = np.mean(list(implicit_individual_knn_ndcg_scores.values()))

        # Collect results including user-item scores, ranks, and NDCG
        implicit_results.append({'Algorithm': 'ALS', 'N': N, 'factors': factors, 'n_nbrs': n_nbrs, 'mean_ndcg': implicit_mean_als_ndcg})
        implicit_results.append({'Algorithm': 'KNN', 'N': N, 'factors': factors, 'n_nbrs': n_nbrs, 'mean_ndcg': implicit_mean_knn_ndcg})

    implicit_results_df = pd.DataFrame(implicit_results)
    print("\nImplicit Data Mean NDCG values:\n")
    print(implicit_results_df)
    
    # Main 
    for threshold_percentage in threshold_percentages:
        pruned_implicit_data = load_and_convert_to_pruned_implicit(implicit_data, threshold_percentage)

        for N, (factors, n_nbrs) in itertools.product(N_values, hyperparameters):
            # Splitting the data and determining the complete set of unique users and items
            all_implicit_pruned_users = pruned_implicit_data['user'].unique()
            all_implicit_pruned_items = pruned_implicit_data['item'].unique()
            implicit_pruned_train_data, implicit_pruned_test_data = train_test_split_data(pruned_implicit_data, train_size=0.8, random_state=42)

            # User-item interaction matrix
            implicit_pruned_train_matrix = create_user_item_matrix(implicit_pruned_train_data, all_implicit_pruned_users, all_implicit_pruned_items)
            
            # Training the Model
            implicit_pruned_als_model = train_als_model(implicit_pruned_train_matrix, factors)
            implicit_pruned_knn_model = train_knn_model(implicit_pruned_train_data, n_nbrs)

            # Generating recommendations for both the models
            implicit_pruned_test_user_ids = implicit_pruned_test_data['user'].unique()
            implicit_pruned_als_recommendations = generate_als_user_item_scores(implicit_pruned_als_model, implicit_pruned_train_matrix, implicit_pruned_test_user_ids, N, filter_already_liked_items=True)
            implicit_pruned_knn_recommendations = generate_knn_recommendations(implicit_pruned_knn_model, implicit_pruned_train_data, implicit_pruned_test_data, N)

            # Individual NDCG scores
            implicit_pruned_individual_als_ndcg_scores = calculate_individual_ndcg(implicit_pruned_als_recommendations, implicit_pruned_test_data, N)                  
            implicit_pruned_individual_knn_ndcg_scores = calculate_individual_ndcg(implicit_pruned_knn_recommendations, implicit_pruned_test_data, N)

            # Mean NDCG score
            implicit_pruned_mean_als_ndcg = np.mean(list(implicit_pruned_individual_als_ndcg_scores.values()))
            implicit_pruned_mean_knn_ndcg = np.mean(list(implicit_pruned_individual_knn_ndcg_scores.values()))
    
            # Collect results including user-item scores, ranks, and NDCG
            pruned_implicit_results.append({'Algorithm': 'ALS', 'N': N, 'factors': factors, 'n_nbrs': n_nbrs, 'Threshold Percentage': f"{int(100 - threshold_percentage*100)}%", 'mean_ndcg': implicit_pruned_mean_als_ndcg})
            pruned_implicit_results.append({'Algorithm': 'KNN', 'N': N, 'factors': factors, 'n_nbrs': n_nbrs, 'Threshold Percentage': f"{int(100 - threshold_percentage*100)}%", 'mean_ndcg': implicit_pruned_mean_knn_ndcg})

    implicit_pruned_results_df = pd.DataFrame(pruned_implicit_results)
    print("\nPruned Implicit Data Mean NDCG values:\n")
    print(implicit_pruned_results_df)

    # Example DataFrame creation (replace with your actual data frame concatenation)
    combined_df = pd.concat([implicit_results_df, implicit_pruned_results_df], ignore_index=True)
    
    als_df = combined_df[combined_df['Algorithm'] == 'ALS'].sort_values(by=['N'])
    als_df.loc[als_df['Algorithm'] == 'ALS', 'n_nbrs'] = "-"
    als_df['Threshold Percentage'] = als_df['Threshold Percentage'].fillna("-")
    als_df = als_df.fillna("-")
    als_df.reset_index(drop=True, inplace=True)
    desired_column_order_als = ['Algorithm', 'N', 'factors', 'n_nbrs', 'Threshold Percentage', 'mean_ndcg']
    als_df= als_df[desired_column_order_als]
    print("\nMean NDCG of pruned and unpruned using ALS Algorithm:")    
    print(als_df)

    knn_df = combined_df[combined_df['Algorithm'] == 'KNN'].sort_values(by=['N'])
    knn_df.loc[knn_df['Algorithm'] == 'KNN', 'factors'] = "-"
    knn_df['Threshold Percentage'] = knn_df['Threshold Percentage'].fillna("-")
    knn_df = knn_df.fillna("-")
    knn_df.reset_index(drop=True, inplace=True)
    desired_column_order_knn = ['Algorithm', 'N', 'factors', 'n_nbrs', 'Threshold Percentage', 'mean_ndcg']
    knn_df= knn_df[desired_column_order_knn]
    print("\nMean NDCG of pruned and unpruned using KNN Algorithm:")
    print(knn_df)
    
    # Combine ALS and KNN dataframes for plotting
    plotting_df = pd.concat([als_df, knn_df], ignore_index=True)
    plotting_df['Factors/Neighbors'] = plotting_df.apply(lambda x: x['factors'] if pd.notnull(x['factors']) else x['n_nbrs'], axis=1)

    # Call the plotting function with the combined DataFrame
    plot_mean_ndcg(plotting_df)

    # Filter for unpruned data
    unpruned_results_df = plotting_df[plotting_df['Threshold Percentage'] == 'Unpruned']
    pruned_results_df = plotting_df[plotting_df['Threshold Percentage'] != 'Unpruned']

    # Calculate the average mean NDCG for ALS and KNN separately
    average_unpruned_als_ndcg = unpruned_results_df[unpruned_results_df['Algorithm'] == 'ALS']['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG for ALS:", average_unpruned_als_ndcg)
    average_unpruned_als_ndcg_N_3 = unpruned_results_df[(unpruned_results_df['Algorithm'] == 'ALS') & (unpruned_results_df['N'] == 3)]['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG at N=3 for ALS:", average_unpruned_als_ndcg_N_3)
    average_unpruned_als_ndcg_N_5 = unpruned_results_df[(unpruned_results_df['Algorithm'] == 'ALS') & (unpruned_results_df['N'] == 5)]['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG at N=5 for ALS:", average_unpruned_als_ndcg_N_5)
    average_unpruned_als_ndcg_N_10 = unpruned_results_df[(unpruned_results_df['Algorithm'] == 'ALS') & (unpruned_results_df['N'] == 10)]['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG at N=10 for ALS:", average_unpruned_als_ndcg_N_10)
    average_unpruned_als_ndcg_N_20 = unpruned_results_df[(unpruned_results_df['Algorithm'] == 'ALS') & (unpruned_results_df['N'] == 20)]['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG at N=20 for ALS:", average_unpruned_als_ndcg_N_20)
    average_pruned_als_ndcg = pruned_results_df[pruned_results_df['Algorithm'] == 'ALS']['mean_ndcg'].mean()
    print("\nAverage Pruned Mean NDCG for ALS:", average_pruned_als_ndcg)
    average_10_percent_pruned_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '10%')]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Mean NDCG for ALS:", average_10_percent_pruned_als_ndcg)
    average_10_percent_pruned_N_3_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '10%') & (pruned_results_df['N'] == 3)]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Mean NDCG at N=3 for ALS:", average_10_percent_pruned_N_3_als_ndcg)
    average_10_percent_pruned_N_5_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '10%') & (pruned_results_df['N'] == 5)]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Mean NDCG at N=5 for ALS:", average_10_percent_pruned_N_5_als_ndcg)
    average_10_percent_pruned_N_10_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '10%') & (pruned_results_df['N'] == 10)]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Mean NDCG at N=10 for ALS:", average_10_percent_pruned_N_10_als_ndcg)
    average_10_percent_pruned_N_20_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '10%') & (pruned_results_df['N'] == 20)]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Mean NDCG at N=20 for ALS:", average_10_percent_pruned_N_20_als_ndcg)
    average_5_percent_pruned_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '5%')]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Meand NDCG for ALS:", average_5_percent_pruned_als_ndcg)
    average_5_percent_pruned_N_3_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '5%') & (pruned_results_df['N'] == 3)]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Mean NDCG at N=3 for ALS:", average_5_percent_pruned_N_3_als_ndcg)
    average_5_percent_pruned_N_5_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '5%') & (pruned_results_df['N'] == 5)]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Mean NDCG at N=5 for ALS:", average_5_percent_pruned_N_5_als_ndcg)
    average_5_percent_pruned_N_10_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '5%') & (pruned_results_df['N'] == 10)]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Mean NDCG at N=10 for ALS:", average_5_percent_pruned_N_10_als_ndcg)
    average_5_percent_pruned_N_20_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '5%') & (pruned_results_df['N'] == 20)]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Mean NDCG at N=20 for ALS:", average_5_percent_pruned_N_20_als_ndcg)
    average_2_percent_pruned_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '2%')]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Meand NDCG for ALS:", average_2_percent_pruned_als_ndcg)
    average_2_percent_pruned_N_3_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '2%') & (pruned_results_df['N'] == 3)]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Mean NDCG at N=3 for ALS:", average_2_percent_pruned_N_3_als_ndcg)
    average_2_percent_pruned_N_5_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '2%') & (pruned_results_df['N'] == 5)]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Mean NDCG at N=5 for ALS:", average_2_percent_pruned_N_5_als_ndcg)
    average_2_percent_pruned_N_10_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '2%') & (pruned_results_df['N'] == 10)]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Mean NDCG at N=10 for ALS:", average_2_percent_pruned_N_10_als_ndcg)
    average_2_percent_pruned_N_20_als_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'ALS') & (pruned_results_df['Threshold Percentage'] == '2%') & (pruned_results_df['N'] == 20)]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Mean NDCG at N=20 for ALS:", average_2_percent_pruned_N_20_als_ndcg)
    average_unpruned_knn_ndcg = unpruned_results_df[unpruned_results_df['Algorithm'] == 'KNN']['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG for KNN:", average_unpruned_knn_ndcg)
    average_unpruned_knn_ndcg_N_3 = unpruned_results_df[(unpruned_results_df['Algorithm'] == 'KNN') & (unpruned_results_df['N'] == 3)]['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG at N=3 for KNN:", average_unpruned_knn_ndcg_N_3)
    average_unpruned_knn_ndcg_N_5 = unpruned_results_df[(unpruned_results_df['Algorithm'] == 'KNN') & (unpruned_results_df['N'] == 5)]['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG at N=5 for KNN:", average_unpruned_knn_ndcg_N_5)
    average_unpruned_knn_ndcg_N_10 = unpruned_results_df[(unpruned_results_df['Algorithm'] == 'KNN') & (unpruned_results_df['N'] == 10)]['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG at N=10 for KNN:", average_unpruned_knn_ndcg_N_10)
    average_unpruned_knn_ndcg_N_20 = unpruned_results_df[(unpruned_results_df['Algorithm'] == 'KNN') & (unpruned_results_df['N'] == 20)]['mean_ndcg'].mean()
    print("\nAverage Unpruned Mean NDCG at N=20 for KNN:", average_unpruned_knn_ndcg_N_20)
    average_pruned_knn_ndcg = pruned_results_df[pruned_results_df['Algorithm'] == 'KNN']['mean_ndcg'].mean()
    print("\nAverage Pruned Mean NDCG for KNN:", average_pruned_knn_ndcg)
    average_10_percent_pruned_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '10%')]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Meand NDCG for KNN:", average_10_percent_pruned_knn_ndcg)
    average_10_percent_pruned_N_3_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '10%') & (pruned_results_df['N'] == 3)]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Mean NDCG at N=3 for KNN:", average_10_percent_pruned_N_3_knn_ndcg)
    average_10_percent_pruned_N_5_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '10%') & (pruned_results_df['N'] == 5)]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Mean NDCG at N=5 for KNN:", average_10_percent_pruned_N_5_knn_ndcg)
    average_10_percent_pruned_N_10_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '10%') & (pruned_results_df['N'] == 10)]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Mean NDCG at N=10 for KNN:", average_10_percent_pruned_N_10_knn_ndcg)
    average_10_percent_pruned_N_20_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '10%') & (pruned_results_df['N'] == 20)]['mean_ndcg'].mean()
    print("\nAverage 10% Pruned Mean NDCG at N=20 for KNN:", average_10_percent_pruned_N_20_knn_ndcg)
    average_5_percent_pruned_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '5%')]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Meand NDCG for KNN:", average_5_percent_pruned_knn_ndcg)
    average_5_percent_pruned_N_3_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '5%') & (pruned_results_df['N'] == 3)]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Mean NDCG at N=3 for KNN:", average_5_percent_pruned_N_3_knn_ndcg)
    average_5_percent_pruned_N_5_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '5%') & (pruned_results_df['N'] == 5)]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Mean NDCG at N=5 for KNN:", average_5_percent_pruned_N_5_knn_ndcg)
    average_5_percent_pruned_N_10_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '5%') & (pruned_results_df['N'] == 10)]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Mean NDCG at N=10 for KNN:", average_5_percent_pruned_N_10_knn_ndcg)
    average_5_percent_pruned_N_20_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '5%') & (pruned_results_df['N'] == 20)]['mean_ndcg'].mean()
    print("\nAverage 5% Pruned Mean NDCG at N=20 for KNN:", average_5_percent_pruned_N_20_knn_ndcg)
    average_2_percent_pruned_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '2%')]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Meand NDCG for KNN:", average_2_percent_pruned_knn_ndcg)
    average_2_percent_pruned_N_3_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '2%') & (pruned_results_df['N'] == 3)]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Mean NDCG at N=3 for KNN:", average_2_percent_pruned_N_3_knn_ndcg)
    average_2_percent_pruned_N_5_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '2%') & (pruned_results_df['N'] == 5)]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Mean NDCG at N=5 for KNN:", average_2_percent_pruned_N_5_knn_ndcg)
    average_2_percent_pruned_N_10_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '2%') & (pruned_results_df['N'] == 10)]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Mean NDCG at N=10 for KNN:", average_2_percent_pruned_N_10_knn_ndcg)
    average_2_percent_pruned_N_20_knn_ndcg = pruned_results_df[(pruned_results_df['Algorithm'] == 'KNN') & (pruned_results_df['Threshold Percentage'] == '2%') & (pruned_results_df['N'] == 20)]['mean_ndcg'].mean()
    print("\nAverage 2% Pruned Mean NDCG at N=20 for KNN:", average_2_percent_pruned_N_20_knn_ndcg)


if __name__ == "__main__":
    main()

