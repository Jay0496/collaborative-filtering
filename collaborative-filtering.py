import pandas as pd
import numpy as np

# Load MovieLens dataset
ml_data = pd.read_csv(
    'https://files.grouplens.org/datasets/movielens/ml-100k/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

# Load Movie Titles
movie_titles = pd.read_csv(
    'https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    names=['item_id', 'title'] + [f'col_{i}' for i in range(22)],
    usecols=['item_id', 'title']
)

# Create a user-item matrix
user_item_matrix = ml_data.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

# Cosine Similarity from Scratch
def cosine_similarity_manual(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / norm
    similarity = normalized_matrix @ normalized_matrix.T
    return similarity

similarity_matrix = cosine_similarity_manual(user_item_matrix.values)

# Recommendation Function with Improved Output
def recommend(user_id, user_item_matrix, similarity_matrix, movie_titles, top_n=5):
    user_index = user_id - 1  # Convert user_id to zero-based index
    similar_users = similarity_matrix[user_index]
    scores = similar_users @ user_item_matrix.values  # Use .values here for matrix multiplication

    # Exclude already-rated items
    rated_items = user_item_matrix.iloc[user_index] > 0  # Use .iloc for the DataFrame
    scores[rated_items.index[rated_items]] = -np.inf  # Mark rated items with negative infinity

    # Get top N recommendations
    recommended_indices = np.argsort(-scores)[:top_n]
    item_ids = user_item_matrix.columns[recommended_indices]  # Use .columns for item IDs
    recommendations = [(item, scores[i]) for i, item in zip(recommended_indices, item_ids)]

    # Map Item IDs to Titles
    recommendations_with_titles = [
        (movie_titles.loc[movie_titles['item_id'] == item_id, 'title'].values[0], score)
        for item_id, score in recommendations
    ]
    return recommendations_with_titles

# Example Recommendations
user_id = 1
recommendations = recommend(user_id, user_item_matrix, similarity_matrix, movie_titles)  # Pass DataFrame here

print(f"Recommendations for User {user_id}:")
for movie_title, score in recommendations:
    print(f"{movie_title} with score {score:.2f}")