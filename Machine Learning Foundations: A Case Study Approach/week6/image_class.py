import turicreate as tc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load a common image analysis dataset
image_train = tc.SFrame('image_train_data/')
image_test = tc.SFrame('image_test_data/')

# Exploring the image data
print(image_train)

# train a classifier on the raw image pixels
raw_pixel_model = tc.logistic_classifier.create(image_train, target='label', features=['image_array'])

# Make a prediction with the simple model based on raw pixels
image_test[0:3]['image'].explore()
print(image_test[0:3]['label'])
print(raw_pixel_model.predict(image_test[0:3]))

# Evaluating raw pixel model on test data
print(raw_pixel_model.evaluate(image_test))

# Can we improve the model using deep features
print(len(image_train))
deep_learning_model = tc.load_model('imagenet_model')
image_train['deep_features'] = deep_learning_model.extract_features(image_train)
print(image_train.head())

# Given the deep features, let's train a classifier
deep_features_model = tc.logistic_classifier.create(image_train,
                                                    features=['deep_features'],
                                                    target='label')

# Apply the deep features model to first few images of test set
print(image_test[0:3]['image'].explore())
print(deep_features_model.predict(image_test[0:3]))

#Compute test_data accuracy of deep_features_model
print(deep_features_model.evaluate(image_test))
