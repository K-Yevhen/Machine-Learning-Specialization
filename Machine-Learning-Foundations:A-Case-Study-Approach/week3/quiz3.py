import turicreate as tc

products = tc.SFrame('amazon_baby.sframe')

# train_data, test_data = products.random_split(.8, seed=0)

products['word_count'] = tc.text_analytics.count_words(products['review'])

def word_count(word_counts, column_name):
    if(column_name not in word_counts):
        return 0
    return word_counts[column_name]

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

for word in selected_words:
    products[word] = products['word_count'].apply(lambda x : word_count(x, word))

# print(products)

for word in selected_words:
    print(word + ':\t' + str(products[word].sum()))
# print(products)

# ignore all 3*  reviews
products = products[products['rating']!= 3]

# positive sentiment = 4-star or 5-star reviews
products['sentiment'] = products['rating'] >= 4

train_data,test_data = products.random_split(.8,seed=0)
selected_words_model = tc.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)

products['predicted_sentiment'] = selected_words_model.predict(products, output_type = 'probability')
# print(selected_words_model.evaluate(test_data))

diaper_champ_reviews = products[products['name']== 'Baby Trend Diaper Champ']
diaper_champ_reviews['predicted_sentiment'] = selected_words_model.predict(diaper_champ_reviews, output_type = 'probability')
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)
print(diaper_champ_reviews.tail(1))