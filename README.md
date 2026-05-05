# Titanic Survival Neural Network

## Description
Machine-learning coursework that trains and evaluates TensorFlow/Keras neural networks on Titanic passenger data. The project includes data splitting, model definition, K-fold training, learning-rate experiments, saved model checkpoints, evaluation, and prediction scripts.

## Tech Stack
- Python
- TensorFlow / Keras
- scikit-learn
- pandas
- NumPy
- Matplotlib

## Structure
- `data_loading.py` prepares train/test data from the Titanic dataset.
- `model_setup.py` defines the dense neural-network architecture.
- `model_train.py` trains models with Stratified K-Fold validation, checkpoints, and early stopping.
- `model_evaluate.py` evaluates a saved model against the test data.
- `make_predicitons.py` creates predictions from a trained model.
- `test_learning_rates.py` compares training behavior across learning rates.
- `IA - Usando gridsearch/` contains an alternate grid-search approach.

## How to Run
The included execution notes list the expected setup:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
python data_loading.py
python model_train.py
python model_evaluate.py
```
