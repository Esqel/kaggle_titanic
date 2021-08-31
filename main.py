import os,sys
import pathlib
from lib.spark import SparkWrapper
from lib.data_transformation import DataTransformation
# Add file dir to PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Globals
DATA_PATH = f"kaggle_titanic/data/train.csv"
# (0, 0.9] range
TRAINING_PERCENTAGE = 0.8

if TRAINING_PERCENTAGE <= 0 or TRAINING_PERCENTAGE > 0.9:
    raise Exception('TRAINING_PERCENTAGE out of (0, 0.9] range!')

with SparkWrapper("titanic_ml_pipeline") as sw:
    data = sw.read_csv(path=DATA_PATH,header=True)

    dt = DataTransformation(data)
    (training_data, test_data) = dt.prepare_data_for_training(
        TRAINING_PERCENTAGE)
    
    model = dt.prepare_model(training_data)
    predictions = model.transform(test_data)
    
    accuracy = dt.evaluate_model(predictions)
    print(f"Model accuracy = {accuracy:.2%}")