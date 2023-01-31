import turicreate as tc

# Load some text data - from wikipedia

people = tc.SFrame("people_wiki.sframe")

print(people.head())
print(len(people))

# Explore the dataset and checkout the text it contains
obama = people[people['name'] == 'Barack Obama']
print(obama)
print(obama['text'])
clooney = people[people['name'] == 'George Clooney']
print(clooney)
print(clooney['text'])

# Get the word counts for Obama article
obama['word_count'] = tc.text_analytics.count_words(obama['text'])
print(obama['word_count'])

# Sort the word counts for Obama article
obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name=['word', 'count'])
print(obama_word_count_table.head())
print(obama_word_count_table.sort('count', ascending=False))

# Compute TF-IDF for the corpus
people['word_count'] = tc.text_analytics.count_words(people['text'])
print(people.head())
people['tfidf'] = tc.text_analytics.tf_idf(people['text'])
print(people)

# Examine the TF-IDF for the Obama article
obama = people[people['name'] == 'Barack Obama']
print(obama[['tfidf']].stack('tfidf', new_column_name=['word', 'tfidf']).sort('tfidf', ascending=False))

# Manually compute distances between a few people
clinton = people[people['name'] == 'Bill Clinton']
beckham = people[people['name'] == 'David Beckham']

# is Obama closer to clinton than to Beckham>
print(tc.distances.cosine(obama['tfidf'][0], clinton['tfidf'][0]))
print(tc.distances.cosine(obama['tfidf'][0], beckham['tfidf'][0]))

# Build a nearest neighbor model for document retrieval
knn_model = tc.nearest_neighbors.create(people, features=['tfidf'], label='name')

# Applying the nearest-neighbors model for retrieval
# Who is closest to obama?
print(knn_model.query(obama))

# Other examples of document retrieval
swift = people[people['name'] == 'Taylor Swift']
print(knn_model.query(swift))
jolie = people[people['name'] == 'Angelina Jolie']
print(knn_model.query(jolie))
arnold = people[people['name'] == 'Arnold Schwarzenegger']
print(knn_model.query(arnold))
