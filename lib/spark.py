from pyspark.sql import SparkSession

class SparkWrapper:
    def __enter__(self):
        return self

    def __init__(self, session_name: str):
        self.session_name = session_name
        self.spark = SparkSession\
            .builder\
            .appName(session_name)\
            .getOrCreate()\

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.spark.stop()
    
    def read_csv(self, path: str, header: bool):
        return self.spark\
            .read\
            .option("header",header)\
            .csv(path)