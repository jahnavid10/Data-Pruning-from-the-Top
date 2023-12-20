# Importing Libraries
import pandas as pd
import numpy as np

# Load the dataset into a Pandas DataFrame
data = pd.read_csv(r'D:/Recommender Systems\Arbeit/ml-100k/ml-100k/u.data', sep =' ', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Split the 'user_id' column on tabs and create new columns
data[['user_id', 'item_id', 'rating', 'timestamp']] = data['user_id'].str.split('\t', expand=True)

# Convert the columns to the appropriate data types
data['user_id'] = data['user_id'].astype(int)
data['item_id'] = data['item_id'].astype(int)
data['rating'] = data['rating'].astype(int)
data['timestamp'] = data['timestamp'].astype(int)

# Print the column names in your dataset
print(data.head())

# Removing Top users
# Calculate total ratings per user
total_user_ratings = data.groupby('user_id')['rating'].count().reset_index()

# Sort user_activity in descending order based on 'total ratings per user'
total_user_ratings = total_user_ratings.sort_values(by='rating', ascending=False)

print(total_user_ratings.head())

# Determine the percentage of top users to prune
pruning_percentage = 0.1 # 10%
num_users_to_prune = int(len(total_user_ratings) * pruning_percentage)

# Identify the top users to prune
top_users_to_prune = total_user_ratings.head(num_users_to_prune)

# Saving Top user data
# Creating a DataFrame using the extracted data
top_user_data = data[data['user_id'].isin(top_users_to_prune['user_id'])]

# Saving the DataFrame as a CSV file
top_user_data.to_csv(r"D:\Recommender Systems\Arbeit\project\top_user.csv", index=False)

# Print the top users
print("Top Users:")
print(top_user_data.head())

# Filter the data to exclude ratings from top users
data = data[~data['user_id'].isin(top_users_to_prune['user_id'])]

# Print the filtered data
print("\nFiltered Data:")
print(data.head())

# Saving the Filtered data (data left after pruning top user data)
# Creating a DataFrame using the extracted data
data = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'timestamp'])

# Saving the DataFrame as a CSV file
data.to_csv(r"D:\Recommender Systems\Arbeit\project\top_pruned.csv", index=False)

# Explicit to Implicit conversion
# Load your explicit data
explicit_data = pd.read_csv(r"D:\Recommender Systems\Arbeit\project\top_pruned.csv")  

# Binarization without threshold
explicit_data['implicit_feedback'] = (explicit_data['rating'] > 0).astype(int)

# Display the modified DataFrame
print(explicit_data[['user_id', 'item_id', 'rating','timestamp']])

# Creating a DataFrame using the extracted data
implicit_data = pd.DataFrame(explicit_data, columns=['user_id', 'item_id', 'rating', 'timestamp', 'implicit_feedback'])

# Saving the DataFrame as a CSV file
implicit_data.to_csv(r"D:\Recommender Systems\Arbeit\project\implicit_data.csv", index=False)

print(implicit_data.head())