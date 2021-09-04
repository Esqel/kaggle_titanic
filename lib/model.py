from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class Model():
    def prepare_model(self, train_df):
        # Use Random Forest Classifier as an estimator
        rf = RandomForestClassifier(
            labelCol="Survived", 
            featuresCol="features",
            maxDepth=5
        )
        return rf.fit(train_df)

    def evaluate_model(self, pred_df):
        evaluator = MulticlassClassificationEvaluator(
            labelCol="Survived", 
            predictionCol="prediction", 
            metricName="accuracy"
        )
        return evaluator.evaluate(pred_df)