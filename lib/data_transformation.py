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

FEATURE_MAPPING = {
    "PassengerId": "int",
    "Survived": "int",
    "Pclass": "int",
    "Sex": "string",
    "Age": "int",
    "Fare": "float",
    "Embarked": "string"
}

class DataTransformation():
    def prepare_data(self, df: DataFrame, data_type: str) -> DataFrame:
        if data_type == "train":
            df = df.select(
                [col(e[0]).cast(e[1]) for e in FEATURE_MAPPING.items()]
            )
        elif data_type == "test":
            df = df.select(
                [
                    col(e[0]).cast(e[1]) 
                    for e in FEATURE_MAPPING.items() 
                    if e[0] != "Survived"
                ]
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