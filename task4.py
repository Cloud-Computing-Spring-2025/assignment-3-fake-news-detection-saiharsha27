import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Fake News Analysis - Task 4") \
        .getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("ERROR")

    print("======= TASK 4: MODEL TRAINING =======")

    # Read the CSV file
    df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

    # Feature extraction pipeline (from Task 2 and 3)
    # Text preprocessing
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    tokenized_df = tokenizer.transform(df)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    cleaned_df = remover.transform(tokenized_df)

    # TF-IDF Vectorization
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
    featurized_df = hashingTF.transform(cleaned_df)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idf_model = idf.fit(featurized_df)
    tfidf_df = idf_model.transform(featurized_df)

    # Label Indexing
    label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
    label_model = label_indexer.fit(tfidf_df)
    indexed_df = label_model.transform(tfidf_df)

    # Split the data into training (80%) and test (20%) sets
    (train_data, test_data) = indexed_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training data size: {train_data.count()}")
    print(f"Test data size: {test_data.count()}")

    # Train Logistic Regression model
    lr = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=10)
    lr_model = lr.fit(train_data)

    # Print some model coefficients information
    print(f"Model coefficients: {lr_model.coefficients}")
    print(f"Model intercept: {lr_model.intercept}")

    # Make predictions on the test set
    predictions = lr_model.transform(test_data)

    # Show sample of predictions
    print("Sample of predictions:")
    predictions.select("id", "title", "label_index", "prediction").show(5, truncate=False)

    # The predictions DataFrame already has simple types (Double) for prediction
    # so no conversion needed for these columns
    
    os.makedirs("output", exist_ok=True)
    
    # Select only the columns we need for output
    predictions_output = predictions.select("id", "title", "label_index", "prediction")
    
    # Write predictions to CSV
    predictions_output.write.csv("output/task4_output", header=True, mode="overwrite")

    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()