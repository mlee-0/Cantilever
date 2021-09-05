# https://www.tensorflow.org/tutorials/keras/text_classification


import os
import re
import shutil
import string

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


if False:
    # Download movie review data. Do not run this after the data has already been downloaded.
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    dataset = tf.keras.utils.get_file('aclImdb_v1', url, untar=True, cache_dir='.', cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')

    # Open a single sample file.
    sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
    with open(sample_file) as f:
        print(f.read())

    # Remove a folder.
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

# Gather datasets.
batch_size = 32
seed = 42
raw_train_dataset = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed,
)
raw_validation_dataset = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed,
)
raw_test_dataset = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size,
)

# Define a custom standardization function that removes HTML tags.
def custom_standardization(input_data):
    input_data = tf.strings.lower(input_data)  # Convert to lowercase
    input_data = tf.strings.regex_replace(input_data, '<br />', ' ')  # Remove HTML tags
    input_data = tf.strings.regex_replace(input_data, '[%s]' % re.escape(string.punctuation), '')  # Remove punctuation
    return input_data

# Create layer used to vectorize the data.
max_features = 10000
sequence_length = 250
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length,
)

# Build an index of strings to integers.
train_text = raw_train_dataset.map(lambda x, y: x)  # Get only the data (no labels)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Show a sample data from the test dataset.
text_batch, label_batch = next(iter(raw_train_dataset))
first_text, first_label = text_batch[0], label_batch[0]
print('Review', first_text)
print('Label', raw_train_dataset.class_names[first_label])
print('Vectorized review', vectorize_text(first_text, first_label))

# Vectorize the datasets.
train_dataset = raw_train_dataset.map(vectorize_text)
validation_dataset = raw_validation_dataset.map(vectorize_text)
test_dataset = raw_test_dataset.map(vectorize_text)

# Configure the datasets for performance.
AUTOTUNE = tf.data.AUTOTUNE
train_datset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model.
embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1),
])
model.summary()

model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0),
)

# Train the model.
epochs = 10
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
)

# Evaluate the model.
loss, accuracy = model.evaluate(test_dataset)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy*100}%')

# Plot how the training metrics changed over time.
history_dict = history.history
accuracy = history_dict['binary_accuracy']
validation_accuracy = history_dict['val_binary_accuracy']
loss = history_dict['loss']
validation_loss = history_dict['val_loss']

epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, validation_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, accuracy, 'ro', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Export the model by first adding the vectorization layer so that it generalizes to non-vectorized data.
model_export = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])
model_export.compile(
    lsos=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)
loss, accuracy = model_export.evaluate(raw_test_dataset)
print(f'Exported model has accuracy: {accuracy*100}%')

# Test the model on new data.
data = [
    '''
    This location is always busy at tech square. Walked in at exact 12 the line was really short but  got long quickly. I usually get the burrito but felt like eating a new item. I had the steak  nachos which was good.
    Veggies were fresh. The meat was tender and the staff was  courteous   There are a lot of other places to eat around here so I don't eat here often.
    Parking  is horrendous. I only have to cOme to the officee once everY two  weeks and I usually work from home. I pay $5 for all day parking at the academy of medicine which is two miles aWay. There are a few spaces in from of the building but they usually aren't available.
    ''',  # Positive
    '''
    I do not recommend coming to this Moe's location. The employees were unfriendly and did not even say welcome to Moe's when we walked in. They took forever to serve us even though there weren't any other customers.
    The restaurant was also a mess. There were cleaning supplies left all over the tables and one of the booth seats had a caution sign left on it, which I almost sat on.
    At one point one of the workers even rudely asked someone to move tables so she could start cleaning early.
    Only reason they got 2 stars instead of 1 is because the food tasted ok.
    ''',  # Negative
    '''
    I went here because I was out of town and craving Chipotle and this was close to my hotel. It is not as good as Chipotle but it did satisfy my craving. This place is also cheaper than Chipotle. My burrito was already closed before I saw that I could put cilantro on there. Darn. Solid 3 stars.
    ''',  # Mixed
]  # https://www.yelp.com/biz/moes-southwest-grill-atlanta-19
model_export.predict(data)