import turicreate as tc

# Loading the CIFAR-10 dataset
# image_test = tc.SFrame('image_test_data')
image_train = tc.SFrame('image_train_data')
# print(image_train)
# print(image_train['image'].explore())
# print(image_train.head())

# Train a nearest-neighbors model for retrieving using deep features
knn_model = tc.nearest_neighbors.create(image_train, features=['deep_features'], label='id')

# Use image retrieval model with deep features to find similar images
cat = image_train[18:19]
# cat['image'].explore()
# print(knn_model.query(cat))

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'], 'id')

cat_neighbours = get_images_from_ids(knn_model.query(cat))
# cat_neighbours.explore()
# input("Enter to exit: ")

car = image_train[8:9]
# car['image'].explore()
# print(get_images_from_ids(knn_model.query(car))['image'].explore())

# Just for fun, let's create a lambda to find and show nearest neighbor images
show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_train[i: i+1]))['image'].explore()

show_neighbors(2000)