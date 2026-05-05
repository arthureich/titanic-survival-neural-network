from sklearn.model_selection import StratifiedKFold
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loading import predictors_train, target_train
i = 0

# Definindo o número de folds para a validação cruzada
n_folds = 5

# Criando o objeto de validação cruzada
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Criando listas para armazenar os resultados de cada fold
fold_train_accuracies = []
fold_test_accuracies = []

# Realizando a validação cruzada
for train_index, test_index in cv.split(predictors_train, np.argmax(target_train, axis=1)):
    # Separando os dados em treinamento e teste para o fold atual
    predictors_train_fold, predictors_test_fold = predictors_train[train_index], predictors_train[test_index]
    target_train_fold, target_test_fold = target_train[train_index], target_train[test_index]
    
    # Criando o modelo usando a função criada anteriormente
    n_cols = predictors_train_fold.shape[1]
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Definindo os callbacks
    checkpoint = ModelCheckpoint('best_model_fold{}.h5'.format(i), save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # Fit no model para o fold
    history = model.fit(predictors_train_fold, target_train_fold, epochs=50, batch_size=32, validation_split=0.2, callbacks=[checkpoint, early_stopping])
    
    # Carregar o melhor modelo salvo durante o treinamento
    model.load_weights('best_model_fold{}.h5'.format(i))
    
    # Validar o modelo para o fold
    train_loss, train_accuracy = model.evaluate(predictors_train_fold, target_train_fold)
    test_loss, test_accuracy = model.evaluate(predictors_test_fold, target_test_fold)
    
    fold_train_accuracies.append(train_accuracy)
    fold_test_accuracies.append(test_accuracy)

    # Incrementando o contador de folds
    i += 1

# Calcular a média e o desvio padrão das acurácias dos folds
mean_train_accuracy = np.mean(fold_train_accuracies)
std_train_accuracy = np.std(fold_train_accuracies)
mean_test_accuracy = np.mean(fold_test_accuracies)
std_test_accuracy = np.std(fold_test_accuracies)

print("Acurácia média no treinamento: {:.2f}% ± {:.2f}%".format(mean_train_accuracy * 100, std_train_accuracy * 100))
print("Acurácia média no teste: {:.2f}% ± {:.2f}%".format(mean_test_accuracy * 100, std_test_accuracy * 100))
