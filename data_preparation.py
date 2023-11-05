# Importing Libraries
import pandas as pd
import numpy as np


# Load the dataset into a Pandas DataFrame
data = pd.read_csv(r'D:\Recommender Systems\Arbeit\ml-100k\ml-100k\u.data', sep =' ', names=['user_id', 'item_id', 'rating', 'timestamp'])

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
# Set a threshold for the number of ratings to define "top users"
ratings_threshold = 500  # Adjust this threshold as needed

# Count the number of ratings for each user
user_ratings_count = data['user_id'].value_counts()

# Get the list of user IDs that exceed the threshold
top_users = user_ratings_count[user_ratings_count > ratings_threshold].index.tolist()

# Filter the data to exclude ratings from top users
data = data[~data['user_id'].isin(top_users)]

print(data.head())
# Set a threshold for implicit feedback (e.g., consider ratings of 4 or 5 as positive)
threshold = 3

# Create a new column 'implicit_feedback' based on the threshold
data['implicit_feedback'] = (data['rating'] >= threshold).astype(int)


# Creating a DataFrame using the extracted data
data = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'timestamp', 'implicit_feedback'])

# Saving the DataFrame as a CSV file
data.to_csv(r"D:\Recommender Systems\Arbeit\project\top_pruned_implicit_feedback.csv", index=False)

print(data.head())