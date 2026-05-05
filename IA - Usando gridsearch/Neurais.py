import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the predictors and target data from the CSV file
data = pd.read_csv('C:\\Users\\Arthur\\Desktop\\IA\\Titanic_dataset.csv', delimiter=',', skiprows=1)

predictors = data.iloc[:, 1:].values  # All columns except the last one are predictors
target = to_categorical(data.iloc[:, 0].values)  # The last column is the target variable 'survived'

# Split the data into training and testing sets
predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.2, random_state=42)

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

print("Data type of predictors_train before conversion:", type(predictors_train))

# Convert predictors_train to a numpy array
predictors_train = predictors_train.astype('float32')

# Verify data type of predictors_train after conversion
print("Data type of predictors_train after conversion:", predictors_train.dtype)
print(target_train.dtype)

# Set up the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=input_shape))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model on the training data
model.fit(predictors_train, target_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(predictors_test, target_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Assuming you have the data for which you want to make predictions in the variable 'pred_data'
# Load the prediction data from the CSV file
pred_data = np.loadtxt('titanic_dataset.csv', delimiter=',')

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:, 1]

# Print predicted_prob_true
print(predicted_prob_true)

# Create list of learning rates: lr_to_test
lr_to_test = [0.0001, 0.001, 0.01]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n' % lr)

    # Build new model to test, unaffected by previous models
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Create Adam optimizer with specified learning rate: my_optimizer
    my_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model on the training data
    model.fit(predictors_train, target_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(predictors_test, target_test)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')