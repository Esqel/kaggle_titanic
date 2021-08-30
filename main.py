from lib.spark import SparkWrapper
from lib.data_transformation import DataTransformation

# Read CSV training data from Kaggle Titanic challenge
DATA_PATH = "/mnt/c/Python//kaggle_titanic/data/train.csv"

with SparkWrapper("titanic_ml_pipeline") as sw:
    data = sw.read_csv(path=DATA_PATH,header=True)
    dt = DataTransformation(data)
    transformed_data = dt.prepare_data_for_training()

    # Split the data into training and test sets
    (training_data, test_data) = transformed_data.randomSplit([0.8,0.2])
    print(
        f"Training data count: {training_data.count()}\n"
        f"Test data count: {test_data.count()}"
        )
    
    model = dt.prepare_model(training_data)
    predictions = model.transform(test_data)
    accuracy = dt.evaluate_model(predictions)
    print(f"Model accuracy = {accuracy:.2%}")