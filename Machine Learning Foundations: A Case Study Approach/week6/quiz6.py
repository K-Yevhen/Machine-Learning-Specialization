import turicreate as tc

image_test = tc.SFrame('image_test_data')
image_train = tc.SFrame('image_train_data')

sketch = tc.Sketch(image_train['label'])
# print(sketch) Question 1

training_data_cat = image_train.filter_by('cat', 'label')
training_data_dog = image_train.filter_by('dog', 'label')
training_data_bird = image_train.filter_by('bird', 'label')
training_data_automobile = image_train.filter_by('automobile', 'label')


cat_model = tc.nearest_neighbors.create(training_data_cat, features=['deep_features'], label='id')
dog_model = tc.nearest_neighbors.create(training_data_dog, features=['deep_features'], label='id')
bird_model = tc.nearest_neighbors.create(training_data_bird, features=['deep_features'], label='id')
automobile_model = tc.nearest_neighbors.create(training_data_automobile, features=['deep_features'], label='id')

cat = image_test[0:1]
# cat['image'].explore()

# Use image retrieval model with deep features to find similar images
def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'], 'id')

cat_neighbors = get_images_from_ids(cat_model.query(cat))
# cat_neighbors['image'].explore()

dog = image_test[0:1]
dog_neighbors = get_images_from_ids(dog_model.query(dog))
# dog_neighbors['image'].explore()

first_image = image_test[0:1]
image_cat_neighbors = cat_model.query(first_image)
# print(image_cat_neighbors)

image_dog_neighbors = dog_model.query(first_image)
# print(image_dog_neighbors, image_cat_neighbors)

image_test_cat = image_test.filter_by('cat', 'label')
image_test_dog = image_test.filter_by('dog', 'label')
image_test_automobile = image_test.filter_by('automobile', 'label')
image_test_bird = image_test.filter_by('bird', 'label')

dog_cat_neighbors = cat_model.query(image_test_dog, k=1)
dog_dog_neighbors = dog_model.query(image_test_dog, k=1)
dog_automobile_neighbors = automobile_model.query(image_test_dog, k=1)
dog_bird_neighbors = bird_model.query(image_test_dog, k=1)

dog_distances = tc.SFrame({'dog-automobile': dog_automobile_neighbors['distance'],
                              'dog-bird': dog_bird_neighbors['distance'],
                              'dog-cat': dog_cat_neighbors['distance'],
                              'dog-dog': dog_dog_neighbors['distance']
                             })

print(dog_distances)

def is_dog_correct(row):
    if row['dog-dog'] <= min(row.values()):
        return 1
    else:
        return 0


print(dog_distances.apply(is_dog_correct).sum())
