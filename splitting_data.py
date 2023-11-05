import pandas as pd
from sklearn.model_selection import train_test_split

# Loading pruned dataset 
data = pd.read_csv(r'D:\Recommender Systems\Arbeit\project\top_pruned_implicit_feedback.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
