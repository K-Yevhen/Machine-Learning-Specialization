import turicreate as tc
import matplotlib

# Load music data
song_data = tc.SFrame("song_data.sframe")

# Explore data
print(song_data.head())
song_data['song'].show()
print(len(song_data))

# Count number of users
users = song_data['user_id'].unique()
print(len(users))

# Create a song recommender
train_data, test_data = song_data.random_split(.8, seed=0)

# Simple popularity-based recommender
popularity_model = tc.popularity_recommender.create(train_data,
                                                    user_id='user_id',
                                                    item_id='song')

# Use the popularity model to make some predictions
print(popularity_model.recommend(users=[users[0]]))
print(popularity_model.recommend(users=[users[1]]))

# Build a song recommender with personalization
personalized_model = tc.item_similarity_recommender.create(train_data,
                                                           user_id='user_id',
                                                           item_id='song')

# Applying the personalized model to make song recommendations
print(personalized_model.recommend(users=[users[0]]))
print(personalized_model.recommend(users=[users[1:4]]))
print(personalized_model.get_similar_items(['With Or Without You - U2']))
print(personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club']))

# Quantitative comparison between the models
model_performance = tc.recommender.util.compare_models(test_data,
                                                       [popularity_model, personalized_model],
                                                       user_sample=0.05)
