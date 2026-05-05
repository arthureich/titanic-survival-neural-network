import numpy as np
from model_setup import model

# Le o CSV 
pred_data = np.loadtxt('titanic_predictions.csv')

# Reshape dos dados (10 samples)
pred_data = pred_data.reshape(1, -1)

# Calcula predictions
predictions = model.predict(pred_data)

# Calcula probabilidade de sobrevivencia
predicted_prob_true = predictions[:, 1]

# Imprime probabilidade
print(predicted_prob_true)
