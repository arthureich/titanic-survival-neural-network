import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from model_setup import model
from data_loading import predictors_train, target_train

# Definindo o número de folds para a validação cruzada
n_folds = 5

# Criando o objeto de validação cruzada
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Criando listas para armazenar os resultados de cada fold
fold_train_accuracies = []
fold_test_accuracies = []

# Realizando a validação cruzada
for i, (train_index, test_index) in enumerate(cv.split(predictors_train, np.argmax(target_train, axis=1))):
    print(f"Fold {i + 1}")
    # Separando os dados em treinamento e teste para o fold atual
    predictors_train_fold, predictors_test_fold = predictors_train[train_index], predictors_train[test_index]
    target_train_fold, target_test_fold = target_train[train_index], target_train[test_index]
    
    # Definindo os callbacks
    checkpoint = ModelCheckpoint(f'best_model_fold{i}.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # Fit no model dos dados de treinamento for para fold atual com callbacks
    history = model.fit(predictors_train_fold, target_train_fold, epochs=50, batch_size=32, validation_split=0.2, callbacks=[checkpoint, early_stopping])
    
    # Carregar o melhor modelo salvo durante o treinamento
    model.load_weights(f'best_model_fold{i}.h5')
    
    # valida o model e testa dados para o fold
    train_loss, train_accuracy = model.evaluate(predictors_train_fold, target_train_fold)
    test_loss, test_accuracy = model.evaluate(predictors_test_fold, target_test_fold)
    
    fold_train_accuracies.append(train_accuracy)
    fold_test_accuracies.append(test_accuracy)

    # Plotar o gráfico de treinamento do modelo
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Model Training - Fold {i + 1}')
    plt.legend()
    plt.savefig(f'model_training_fold{i}.png')
    plt.close()

# Calcular a média e o desvio padrão das acurácias dos folds
mean_train_accuracy = np.mean(fold_train_accuracies)
std_train_accuracy = np.std(fold_train_accuracies)
mean_test_accuracy = np.mean(fold_test_accuracies)
std_test_accuracy = np.std(fold_test_accuracies)

print("Acurácia média no treinamento: {:.2f}% ± {:.2f}%".format(mean_train_accuracy * 100, std_train_accuracy * 100))
print("Acurácia média no teste: {:.2f}% ± {:.2f}%".format(mean_test_accuracy * 100, std_test_accuracy * 100))

# Gráfico de comparação das acurácias dos folds
plt.figure(figsize=(8, 6))
plt.bar(range(1, n_folds + 1), fold_train_accuracies, label='Train Accuracy')
plt.bar(range(1, n_folds + 1), fold_test_accuracies, label='Test Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Fold-wise Accuracy Comparison')
plt.legend()
plt.xticks(np.arange(1, n_folds + 1))
plt.savefig('accuracy_comparison.png')
plt.show()
