# https://www.tensorflow.org/tutorials/keras/regression


# Install on Colab.
# !pip install -q seaborn


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


# Plotting helper functions.
def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

def plot_horsepower(x, y):
    plt.figure()
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()

# Download data.
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
# Clean the data.
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# Split the data into training and testing.
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# Split datasets into features and labels.
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Normalize the data.
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
normalizer.mean.numpy()

# Build the model.
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = preprocessing.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    tf.keras.layers.Dense(units=1),
])

# Compile the model for training and train it.
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error',
)
history = horsepower_model.fit(train_features['Horsepower'], train_labels, epochs=100, verbose=False, validation_split=0.2)
history_data = pd.DataFrame(history.history)
history_data['epoch'] = history.epoch
plot_loss(history)

# Test the model.
test_results = {}
test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels,
    verbose=False,
)
x = tf.linspace(0, 250, 251)
y = horsepower_model.predict(x)
plot_horsepower(x, y)

# Build, train, and test the multiple-inputs model.
linear_model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(units=1),
])
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error',
)
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=False,
    validation_split=0.2,
)
plot_loss(history)
test_results['linear_model'] = linear_model.evaluate(test_features, test_labels, verbose=False)

# Build DNN models.
def build_and_compile_model(normalizer):
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    return model

# Build, train, and test a single-input DNN model.
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
history = dnn_horsepower_model.fit(train_features['Horsepower'], train_labels, epochs=100, verbose=False, validation_split=0.2)
plot_loss(history)
x = tf.linspace(0, 250, 251)
y = dnn_horsepower_model.predict(x)
plot_horsepower(x, y)
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(test_features['Horsepower'], test_labels, verbose=False)

# Build, train, and test a multiple-input DNN model.
dnn_model = build_and_compile_model(normalizer)
history = dnn_model.fit(train_features, train_labels, epochs=100, verbose=False, validation_split=0.2)
plot_loss(history)
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=False)

# Show all test results.
print(pd.DataFrame(test_results, index=['Mean Absolute Error [MPG]']).T)

# Predict using the model.
test_predictions = dnn_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')