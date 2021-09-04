import os
from lib.spark import SparkWrapper
from lib.data_transformation import DataTransformation
from lib.model import Model

# Globals
TRAIN_DATA_PATH = f"{os.path.dirname(__file__)}/data/train.csv"
TEST_DATA_PATH = f"{os.path.dirname(__file__)}/data/test.csv"
TRAIN_PERCENT = 0.8

if __name__ == "__main__":
    if TRAIN_PERCENT <= 0 or TRAIN_PERCENT > 0.9:
        raise Exception('TRAIN_PERCENT out of (0, 0.9] range!')

    with SparkWrapper("titanic_ml_pipeline") as sw:
        train_data = sw.read_csv(path=TRAIN_DATA_PATH, header=True)
        test_data = sw.read_csv(path=TEST_DATA_PATH, header=True)

        dt = DataTransformation()
        train_df = dt.prepare_data(train_data, "train")
        test_df = dt.prepare_data(test_data, "test")

        model = Model()
        rf = model.prepare_model(train_df)
        # print(rf.explainParams())
        predictions = rf.transform(test_df)