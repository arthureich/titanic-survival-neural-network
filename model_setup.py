import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from data_loading import predictors_train, target_train

# Salva o numero de colunas em predictors: n_cols
n_cols = predictors_train.shape[1]

# Criando modelo de rede neural com regularização L2
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(n_cols,), kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(2, activation='softmax'))

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
