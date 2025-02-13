# Collaborative Filtering Recommendation System

This project implements a **user-based collaborative filtering** recommendation system using the **MovieLens 100k dataset**. The primary goal is to recommend movies to users based on the preferences of others with similar tastes.

---

## Why I Made This

1. **Apply Collaborative Filtering to Other Projects**: This project serves as a foundation for integrating collaborative filtering techniques into other domains, such as product recommendation or personalized content delivery.
2. **Enhance Understanding of Recommendation Systems**: To gain practical experience in implementing a recommendation system and understanding its core mathematical principles.
3. **Build Skills with Real-World Datasets**: Work with the MovieLens dataset, a benchmark dataset in recommender systems research.
4. **Learn Key Libraries and Tools**: Practice using libraries like `pandas` and `numpy`, as well as implementing algorithms manually for better understanding.

---

## What I Learned

1. **Cosine Similarity**:
   - Learned how to compute cosine similarity manually, understanding how it measures the similarity between two vectors based on their angles.
2. **Collaborative Filtering**:
   - Gained insights into user-based collaborative filtering and how to aggregate preferences from similar users.
3. **Handling Sparse Data**:
   - Learned to handle missing data by filling gaps appropriately in a user-item matrix.
4. **Dataset Manipulation**:
   - Practiced creating pivot tables, normalizing data, and using metadata to enhance outputs.
5. **Real-World Problem Solving**:
   - Addressed challenges like excluding already-rated items from recommendations and improving output readability.

---

## What the Code Does

1. **Loads the Dataset**:
   - Imports the MovieLens 100k dataset, including both the ratings data and movie metadata (titles).

2. **Creates a User-Item Matrix**:
   - Builds a matrix where rows represent users, columns represent items (movies), and values are ratings.

3. **Computes Cosine Similarity**:
   - Calculates user-user similarity manually based on their rating vectors.

4. **Generates Recommendations**:
   - Aggregates ratings from similar users to generate scores for items.
   - Filters out items the user has already rated.
   - Outputs recommendations with movie titles and relevance scores.

5. **Enhanced Output**:
   - Maps recommended item IDs to movie titles for a user-friendly result.

---

## Features

- Implements **collaborative filtering** from scratch.
- Uses the **MovieLens 100k dataset** for demonstration.
- Outputs movie recommendations with titles and scores.

---

## How It Works
### Key Steps

1. **Compute Similarity Matrix**:
   - Normalizes user vectors and calculates pairwise dot products.
2. **Score Calculation**:
   - Multiplies the similarity vector for the target user by the user-item matrix to compute scores.
3. **Recommendation**:
   - Filters out items already rated by the user.
   - Sorts items by score and retrieves the top results.

---

## Usage

### Prerequisites
Install the required libraries:
```bash
pip install pandas numpy
```

### Running the Code
Run the script to compute recommendations:
```bash
python collaborative_filtering.py
```

### Output Example
```plaintext
Recommendations for User 1:
- "Toy Story (1995)" with score 4.56
- "GoldenEye (1995)" with score 3.87
- "Heat (1995)" with score 3.75
```

---

## Dataset Information

The **MovieLens 100k dataset** contains:
- 100,000 ratings (1-5) from 943 users on 1,682 movies.
- Metadata for movies, such as titles and genres.
- [Dataset Metadata](https://files.grouplens.org/datasets/movielens/ml-100k/u.item).

---

## Future Improvements

1. **Item-Based Collaborative Filtering**:
   - Explore item-item similarity for recommendations.
2. **Content-Based Filtering**:
   - Incorporate features like genres and directors.
3. **Hybrid Methods**:
   - Combine collaborative and content-based approaches for better accuracy.
4. **Interactive Feedback**:
   - Allow users to rate recommendations for iterative improvement.

---

## Credits

- Dataset provided by [GroupLens](https://grouplens.org/).
- Project inspired by collaborative filtering concepts in recommender systems.
