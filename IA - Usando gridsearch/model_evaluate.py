from model_setup import model
from data_loading import predictors_test, target_test
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

# Le o CSV
test_data = pd.read_csv('test_data.csv')

# Separa predictors e target 
target_test = test_data['survived'].values
predictors_test = test_data.drop('survived', axis=1).values

print(target_test)
# Valida o modelo
test_loss, test_accuracy = model.evaluate(predictors_test, to_categorical(target_test))
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Faz predictions
predictions = model.predict(predictors_test)

# Salva os predictions para o CSV
predicted_prob_true = predictions[:10, 1]

# Faz o reshape
predicted_prob_true = predicted_prob_true.reshape(-1, 1)

# Salva as probabilidades em 'titanic_predictions.csv' 
np.savetxt('titanic_predictions.csv', predicted_prob_true, delimiter=',')