import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Loading and exploring product review data

products = tc.SFrame("amazon_baby.sframe")

print(products.head())

# Build the word count vector for each review
products['word_count'] = tc.text_analytics.count_words(products['review'])
print(products)

products['name'].show()
# input("Enter to quit")

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']
print(len(giraffe_reviews))
giraffe_reviews['rating'].show()

# Build a sentiment classifier
products['rating'].show()

# Define what's a positive and a negative sentiment
# ignore all 3* reviews
products = products[products['rating'] != 3]

positive sentiment = 4* and 5* reviews
products['sentiment'] = products['rating'] >= 4
print(products.head())

# Let's train the sentiment classifier
train_data, test_data = products.random_split(.8, seed=0)
sentiment_model = tc.logistic_classifier.create(train_data,
                                                target='sentiment',
                                                features=['word_count'],
                                                validation_set=test_data)

print(sentiment_model.evaluate(test_data, metric='roc_curve'))

# Applying the learned model to understand sentiment the Giraffe
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')
print(giraffe_reviews.head())

# Sort the reviews based on the predicted sentiment and explore
giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
print(giraffe_reviews.head())

print(giraffe_reviews[0]['review'])
print(giraffe_reviews[1]['review'])

# Show most negative reviews
print(giraffe_reviews[-1]['review'])
print(giraffe_reviews[-2]['review'])
