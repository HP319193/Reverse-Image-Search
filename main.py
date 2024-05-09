import tensorflow as tf
import numpy as np
import os
from annoy import AnnoyIndex
import cv2
import argparse

parser = argparse.ArgumentParser(description='Description of your script')

# Add arguments
parser.add_argument('--database_dir_path', type=str, help='Directory path of database images', default='database')
parser.add_argument('--query_image_path', type=str, default='query/sample.jpg', help='Query image file path')

# Parse the arguments
args = parser.parse_args()

database_directory = args.database_dir_path
query_image_path = args.query_image_path

efficientnet_model = tf.keras.applications.EfficientNetB4(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    efficientnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

def extract_features(image):
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    # Extract features using the model
    features = model.predict(img)
    # Apply L2 normalization to the features
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return features

database_features = []
database_filenames = []
for filename in os.listdir(database_directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(database_directory, filename)
        features = extract_features(image_path)
        database_features.append(features.flatten())  # Flatten the features
        database_filenames.append(filename)

num_trees = 200  # Adjust this parameter for better accuracy
feature_dim = len(database_features[0])

annoy_index = AnnoyIndex(feature_dim, 'angular')  # Use cosine similarity

for i, feature in enumerate(database_features):
    annoy_index.add_item(i, feature)

annoy_index.build(num_trees)

num_items = annoy_index.get_n_items()

query_features = extract_features(query_image_path).flatten()

print(query_features)
# Define the number of similar images to retrieve
top_n = 3
# Find the top N similar images
similar_indices = annoy_index.get_nns_by_vector(query_features, top_n)

# print(similar_indices)
query_image = cv2.imread(query_image_path)

for i, index in enumerate(similar_indices):
    similar_image_filename = database_filenames[index]
    similar_image_path = os.path.join(database_directory, similar_image_filename)
    similar_image = cv2.imread(similar_image_path)
    # Calculate the distance manually
    distance = np.linalg.norm(query_features - database_features[index])
    print(f"Similar Image {i + 1}: Distance - {distance}")
    cv2.imshow(f"Similar Image {i + 1}", similar_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


