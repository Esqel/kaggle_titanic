from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler

REQUIRED_FEATURES = [
    "Pclass",
    "Age",
    "Fare",
    "Gender",
    "Boarded"
]

class DataTransformation():
    def prepare_data(self, df: DataFrame, data_type: str):
        if data_type == "train":
            df = df.select(
                col("Survived").cast("int"),
                col("Pclass").cast("int"),
                col("Sex"),
                col("Age").cast("int"),
                col("Fare").cast("float"),
                col("Embarked")
            )
        elif data_type == "test":
            df = df.select(
                col("Pclass").cast("int"),
                col("Sex"),
                col("Age").cast("int"),
                col("Fare").cast("float"),
                col("Embarked")
            )
        else:
            print("Wrong data type chosen!")
            exit

        df = df.replace("?", None).dropna(how="any")
        # Encode two feature columns as numbers
        df = StringIndexer(
            inputCol="Sex", 
            outputCol="Gender", 
            handleInvalid="keep"
        ).fit(df).transform(df)
        df = StringIndexer(
            inputCol="Embarked", 
            outputCol="Boarded", 
            handleInvalid="keep"
        ).fit(df).transform(df)
        
        # Drop unnecessary columns
        df = df.drop("Sex")
        df = df.drop("Embarked")

        # Assemble all the features with VectorAssembler
        assembler = VectorAssembler(
            inputCols=REQUIRED_FEATURES,
            outputCol="features"
        )

        return assembler.transform(df)