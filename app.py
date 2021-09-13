import os
from pyspark.sql.functions import col
# from flask import Flask
from lib.spark import SparkWrapper
from lib.data_transformation import DataTransformation
from lib.model import Model

# Globals
TRAIN_DATA_PATH = f"{os.path.dirname(__file__)}/data/train.csv"
TEST_DATA_PATH = f"{os.path.dirname(__file__)}/data/test.csv"
EVAL_DATA_PATH = f"{os.path.dirname(__file__)}/data/gender_submission.csv"
TRAIN_PERCENT = 0.8
DEFAULT_APP_RESPONSE = "Model is still working"

# app = Flask("test")

if __name__ == "__main__":
    if TRAIN_PERCENT <= 0 or TRAIN_PERCENT > 0.9:
        raise Exception('TRAIN_PERCENT out of (0, 0.9] range!')

    with SparkWrapper("titanic_ml_pipeline") as sw:
        model = Model()
        dt = DataTransformation()

        # @app.route("/result")
        # async def print_result(result = DEFAULT_APP_RESPONSE) -> str:
        #     result = await print("test")
        #     print("Returning model accuracy")
        #     return result

        # app.run(host="0.0.0.0",debug=True,port=5000)
        # print("Doing something")

        train_data = sw.read_csv(path=TRAIN_DATA_PATH, header=True)
        test_data = sw.read_csv(path=TEST_DATA_PATH, header=True)
        eval_data = sw.read_csv(path=EVAL_DATA_PATH, header=True)

        train_df = dt.prepare_data(train_data, "train")
        test_df = dt.prepare_data(test_data, "test")

        rf = model.prepare_model(train_df)
        predictions = rf.transform(test_df)

        result_df = predictions.join(eval_data, "PassengerId", "inner")\
            .select("PassengerId", col("Survived").cast("int"), "prediction")
        # print_result(model.get_model_accuracy(result_df))