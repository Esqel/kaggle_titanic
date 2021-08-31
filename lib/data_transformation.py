from numpy import arange
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class DataTransformation(DataFrame):
    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)

    def prepare_data_for_training(
                                self, 
                                train_percentage: float
                                ):
        result = ()

        if train_percentage > 0.0 and train_percentage <= 0.9:
            df = self.select(
                                col("Survived").cast("float"),
                                col("Pclass").cast("float"),
                                col("Sex"),
                                col("Age").cast("float"),
                                col("Fare").cast("float"),
                                col("Embarked")
                            )
            df = df.replace("?", None) \
                .dropna(how="any")
            # Encode two feature columns as numbers
            df = StringIndexer(
                                inputCol="Sex", 
                                outputCol="Gender", 
                                handleInvalid="keep") \
                                .fit(df) \
                                .transform(df)
            df = StringIndexer(
                                inputCol="Embarked", 
                                outputCol="Boarded", 
                                handleInvalid="keep") \
                                .fit(df) \
                                .transform(df)
            
            # Drop unnecessary columns
            df = df.drop("Sex")
            df = df.drop("Embarked")

            # Assemble all the features with VectorAssembler
            required_features = ["Pclass",
                                "Age",
                                "Fare",
                                "Gender",
                                "Boarded"
                            ]
            assembler = VectorAssembler(
                                        inputCols=required_features,
                                        outputCol="features"
                                    )
            
            result = assembler.transform(df)\
                .randomSplit(
                            [train_percentage,
                            1 - train_percentage]
                            )
            print(
                f"Training data count: {result[0].count()}\n"
                f"Test data count: {result[1].count()}"
                )
        else:
            print("Training percentage is not in (0, 0.9] range!")
        return result

        
    
    def prepare_model(self, training_data):
        # Use Random Forest Classifier as an estimator
        rf = RandomForestClassifier(
                                    labelCol="Survived", 
                                    featuresCol="features",
                                    maxDepth=5
                                )
        return rf.fit(training_data)

    def evaluate_model(self, predictions):
        evaluator = MulticlassClassificationEvaluator(
                                                    labelCol="Survived", 
                                                    predictionCol="prediction", 
                                                    metricName="accuracy"
                                                )
        return evaluator.evaluate(predictions)