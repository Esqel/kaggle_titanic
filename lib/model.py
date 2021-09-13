from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.dataframe import DataFrame

class Model():
    def prepare_model(self, train_df) -> RandomForestClassifier:
        # Use Random Forest Classifier as an estimator
        rf = RandomForestClassifier(
            labelCol="Survived", 
            featuresCol="features",
            maxDepth=5
        )
        return rf.fit(train_df)

    def evaluate_model(self, pred_df) -> MulticlassClassificationEvaluator:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="Survived", 
            predictionCol="prediction", 
            metricName="accuracy"
        )
        return evaluator.evaluate(pred_df)
    
    def get_model_accurracy(self, result_df: DataFrame) -> str:
        return f"Model accuracy: \
            {'{:.1%}'.format(self.evaluate_model(result_df))}"
    
    # WIP
    def bootstrap_model(self, train_df):
        for i in range(0,10):
            (boot_train_df, boot_test_df) = tuple(
                train_df.randomSplit([0.8, 0.2])
            )
            (model, evaluator) = (
                self.prepare_model(), 
                self.evaluate_model(boot_test_df)
            )

