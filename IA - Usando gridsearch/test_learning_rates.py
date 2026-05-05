from sklearn.model_selection import StratifiedKFold
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loading import predictors_train, target_train


i = 0
n_cols = predictors_train.shape[1]

# Função para criar o modelo da rede neural
def create_model(neurons_layer1=64, neurons_layer2=32, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation='relu', input_shape=(n_cols,), kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dense(neurons_layer2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dense(2, activation='softmax'))
    
    # Compilar o modelo com o otimizador e taxa de aprendizado fornecidos
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Criar o modelo KerasClassifier para uso com o GridSearchCV
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)

# Definir a grade de hiperparâmetros para pesquisa
param_grid = {
    'neurons_layer1': [32, 64, 128],
    'neurons_layer2': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Criar o objeto GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

# Executar o GridSearch
grid_result = grid.fit(predictors_train, target_train)

# Imprimir os resultados
print("Melhores hiperparâmetros encontrados:")
print(grid_result.best_params_)
print("Melhor acurácia média durante a validação cruzada: {:.2f}%".format(grid_result.best_score_ * 100))

