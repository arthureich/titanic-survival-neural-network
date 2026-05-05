import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("Carregando os dados do arquivo CSV")
# Carrega os dados do CSV
data = pd.read_csv('Titanic_dataset.csv', delimiter=',', skiprows=1)

print("Dados carregados com sucesso!")

# Separa a primeira coluna das outras
target = to_categorical(data.iloc[:, 0].values)  # Primeira coluna 'survived' é o target
predictors = data.iloc[:, 1:].values  # Outras são predictors

# Converte dados para float
predictors = predictors.astype('float32')

target_one_hot = to_categorical(target)

# Divide os dados em treino e teste
predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.2, random_state=42)

# Salva os dados de treinamento no CSV
train_data = np.column_stack((target_train.argmax(axis=1), predictors_train))  # Combina target e predictors
columns = ['survived'] + list(data.columns[1:])
train_df = pd.DataFrame(train_data, columns=columns)  # Converte para DataFrame com os nomes das colunas
train_df.to_csv('train_data.csv', index=False)
