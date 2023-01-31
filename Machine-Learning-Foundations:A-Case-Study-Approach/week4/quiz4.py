import turicreate as tc

people = tc.SFrame('people_wiki.sframe')

elton = people[people['name'] == 'Elton John']

elton['word_count'] = tc.text_analytics.count_words(elton['text'])
elton_word_count_table = elton[['word_count']].stack('word_count', new_column_name=['word', 'count'])
# print(elton_word_count_table.sort('count', ascending=False))

people['word_count'] = tc.text_analytics.count_words(people['text'])
people['tfidf'] = tc.text_analytics.tf_idf(people['text'])
elton = people[people['name'] == 'Elton John']
# print(elton[['tfidf']].st# ack('tfidf', new_column_name=['word', 'tfidf']).sort('tfidf', ascending=False))

# knn_model = tc.nearest_neighbors.create(people, features=['tfidf'], label='name')
victoria = people[people['name'] == 'Victoria Beckham']
paul = people[people['name'] == 'Paul McCartney']
# print(tc.distances.cosine(elton['tfidf'][0], victoria['tfidf'][0]))
# print(tc.distances.cosine(elton['tfidf'][0], paul['tfidf'][0]))

knn_model = tc.nearest_neighbors.create(people, features=['word_count'], label='name', distance='cosine')
# print(knn_model.query(elton.head(10)))
print(knn_model.query(victoria))

