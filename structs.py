from pyspark.sql.types import *

heart_schema = StructType((StructField('index',IntegerType(),True), \
    StructField('biking',DoubleType(),True), \
        StructField('smoking',DoubleType(),True), \
            StructField('heart_disease',DoubleType(),True)))