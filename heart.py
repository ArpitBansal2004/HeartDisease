from pyspark.sql import SparkSession
from structs import heart_schema
spark = SparkSession.builder.getOrCreate()

heart_df = spark.read.csv('heart_data.csv', header = True, schema = heart_schema)

#heart_df.printSchema()
#print(heart_df.schema)

heart_df.show()

heart_df.createOrReplaceTempView('HEART_DATA')
df = spark.sql('SELECT * FROM HEART_DATA')
df = df.drop('heart_disease')
df = df.drop('index')
df.printSchema()

inputlist = df.columns
print(inputlist)


from pyspark.ml.feature import VectorAssembler

features_assembler = VectorAssembler(inputCols = inputlist, outputCol = 'features')

df = features_assembler.transform(heart_df)

df.printSchema()

working_df = df.select('features', 'heart_disease')
working_df.show()

training, test = working_df.randomSplit([0.7, 0.3])
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol='features', labelCol='heart_disease')

model = lr.fit(training)
predict_output = model.transform(test)

predict_output.show()