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
import seaborn as sns

# Function for Loading and preprocess data
def load_and_convert_to_implicit(file_path, separator=','):
    # Loading data with specified column names and separator
    data = pd.read_csv(file_path, sep=separator, header=0)

    # Convert the columns to the appropriate data types
    data['user'] = data['user'].astype(int)
    data['item'] = data['item'].astype(int)
    
    # Check if a 'rating' column exists in the data
    if 'rating' in data.columns:
        # If rating column exists, assign 1 to imp_rating if rating is present, else 0
        data['imp_rating'] = data['rating'].apply(lambda x: 1 if pd.notnull(x) else 0)
    else:
        # If no rating column, assume all interactions are positive and set imp_rating to 1
        data['imp_rating'] = 1

    columns_to_keep = ['user', 'item', 'imp_rating']
    implicit_data = data[columns_to_keep]
    
    return implicit_data

# Pruning the Implicit data 
def load_and_convert_to_pruned_implicit(implicit_data, threshold_percentage):
    # Pruning the top users based on the number of interactions
    user_interactions = implicit_data.groupby('user').size()
    threshold = user_interactions.quantile(threshold_percentage)
    top_users = user_interactions[user_interactions.values >= threshold].index
    pruned_implicit_data = implicit_data[~implicit_data['user'].isin(top_users)]


    return pruned_implicit_data

# Splitting the data into train and test data
def train_test_split_data(df, train_size=0.8, random_state=42):
    # Sorting user column maintain consistency
    df = df.sort_values(by='user').reset_index(drop=True)

    # Split each user's data to ensure they appear in both train and test sets
    train_rows = []
    test_rows = []
    for _, group in df.groupby('user'):
        n = len(group)
        if n > 1: # users with at least 2 interactions
            train_count = max(int(n * train_size), 1) 
            train_indices = group.sample(n=train_count, random_state=random_state).index.tolist()
            test_indices = list(set(group.index) - set(train_indices))
     
            # Ensure both lists get extended only if there are at least 2 interactions
            train_rows.extend(train_indices)
            test_rows.extend(test_indices)

    # Extract train and test sets
    train_data = df.loc[train_rows].sort_values(by=['user', 'item']).reset_index(drop=True)
    test_data = df.loc[test_rows].sort_values(by=['user', 'item']).reset_index(drop=True)
    
    return train_data, test_data

# Function to create a user-item interaction matrix
def create_user_item_matrix(df, all_users, all_items):
    # Determine the complete set of users and items
    all_users = df['user'].unique()
    all_items = df['item'].unique()
    # Create a sparse matrix where rows represent users, columns represent items
    user_item_matrix = csr_matrix((df['imp_rating'], (df['user'], df['item'])), shape=(max(all_users) + 1, max(all_items) + 1))
    return user_item_matrix

# Function to train ALS model
def train_als_model(train_matrix, factors):
    als_model = AlternatingLeastSquares(factors, regularization=0.1, iterations=20)
    als_model.fit(train_matrix)
    return als_model

# Function to train KNN model 
def train_knn_model(train_data, n_nbrs):
    # Initializing the ItemItem model with implicit feedback
    implicit_knn = ItemItem(n_nbrs, feedback='implicit')
    knn_model = Recommender.adapt(implicit_knn)
    knn_model.fit(train_data)
    return knn_model

# Function for generating als user-item scores and ranks
def generate_als_user_item_scores(model, user_item_matrix, users, N, filter_already_liked_items=True):
    user_item_scores = []
    for user in users:
        user_row = user_item_matrix.getrow(user)
        recommendations = model.recommend(user, user_row, N=N, filter_already_liked_items=filter_already_liked_items)
        user_item_scores.extend([(user, item, score) for item, score in zip(recommendations[0], recommendations[1])])
    user_item_scores_df = pd.DataFrame(user_item_scores, columns=['user', 'item', 'score'])
    user_item_scores_df['rank'] = user_item_scores_df.groupby('user')['score'].rank(ascending=False, method='first')
    return user_item_scores_df

# Function for generating knn user-item scores and ranks
def generate_knn_recommendations(model, train_data, test_data, N):
    unique_items = train_data['item'].unique()
    test_users = test_data['user'].unique()
    all_recs = []

    for user in test_users:
        # Get items that the user has interacted with
        interacted_items = train_data[train_data['user'] == user]['item'].unique()
        # Filter out interacted items from all items
        non_interacted_items = np.setdiff1d(unique_items, interacted_items)

        # Generate recommendations for non-interacted items
        user_recs = model.recommend(user, N, candidates=non_interacted_items)
        for rank, (item, score) in enumerate(zip(user_recs['item'], user_recs['score']), 1):
            all_recs.append({'user': user, 'item': item, 'score': score, 'rank': rank})

    return pd.DataFrame(all_recs, columns=['user', 'item', 'score', 'rank'])

# Function to calculate the individual NDCG and Mean NDCG values for both the algorithms
def calculate_individual_ndcg(df_test, df_true, N):
    ndcg_scores = {}
    for user in df_test['user'].unique():
        true_items = df_true[df_true['user'] == user]['item']
        predicted_items = df_test[df_test['user'] == user].nlargest(N, 'score')['item']

        relevance = [1 if item in true_items.values else 0 for item in predicted_items.values]
        relevance += [0] * (N - len(relevance))
        ndcg_scores[user] = ndcg_score([relevance], [np.arange(1, N+1)])

    return ndcg_scores

# Function to plot NDCG
def plot_mean_ndcg(plotting_df):
    # Replace '-' with 'Unpruned' for plotting
    plotting_df['Threshold Percentage'] = plotting_df['Threshold Percentage'].replace('-', 'Unpruned')

    # Defining the color palette
    palette = {
        'Unpruned': 'tab:green',
        '2%': 'tab:blue',
        '5%': 'tab:orange',
        '10%': 'tab:red'
    }

    hue_order = ['Unpruned', '10%', '5%', '2%']

    # Set up the plot
    N_values = sorted(plotting_df['N'].unique())
    algorithms = ['ALS', 'KNN']
    fig, axes = plt.subplots(nrows=len(N_values), ncols=len(algorithms), figsize=(15, 5 * len(N_values)), sharey=True)
    # Finding the global max NDCG value for setting y-axis limit
    global_max_ndcg = plotting_df['mean_ndcg'].max()
    y_limit_upper = global_max_ndcg + 0.2

    for i, N in enumerate(N_values):
        for j, alg in enumerate(algorithms):
            ax = axes[i][j]
            subset = plotting_df[(plotting_df['N'] == N) & (plotting_df['Algorithm'] == alg)]
            sns.barplot(ax=ax, data=subset, x='factors' if alg == 'ALS' else 'n_nbrs', y='mean_ndcg', hue='Threshold Percentage', palette=palette, hue_order=hue_order)
            ax.set_title(f'{alg} (N={N})')
            ax.set_xlabel('Factors' if alg == 'ALS' else 'Neighbors')
            ax.set_ylabel(f'Mean NDCG at N={N}' if j == 0 else '')
            # Setting y-axis scale
            ax.set_ylim(0, y_limit_upper)
            ax.set_yticks(np.arange(0, y_limit_upper, 0.1))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Adding a main title and adjusting the space at the top
    plt.subplots_adjust(top=0.8)
    plt.suptitle('Movielens-100k', fontsize=20)
    fig.text(0.5, 0.85, 'Mean NDCG Comparison for ALS and KNN by N Value', ha='center', fontsize=20)

    # Create a unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=len(labels))

    # Remove individual legends to avoid duplication
    for ax in axes.flat:
        ax.get_legend().remove()

    # Save the figure with a timestamp
    folder_path = '/home/g057635/arbeit/plots/'
    plt.savefig(f'{folder_path}Movielens_100k.png', bbox_inches='tight')

    plt.show()