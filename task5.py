import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Fake News Analysis - Task 5") \
        .getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("ERROR")

    print("======= TASK 5: EVALUATE THE MODEL =======")

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

    # Split the data (from Task 4)
    (train_data, test_data) = indexed_df.randomSplit([0.8, 0.2], seed=42)

    # Train the model (from Task 4)
    lr = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=10)
    lr_model = lr.fit(train_data)

    # Make predictions
    predictions = lr_model.transform(test_data)

    # Evaluate the model
    # 1. Accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index", 
        predictionCol="prediction",
        metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    # 2. F1 Score
    evaluator.setMetricName("f1")
    f1_score = evaluator.evaluate(predictions)

    # Print metrics
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # Create a DataFrame with the metrics for easier saving
    metrics_data = [("Accuracy", float(accuracy)), ("F1 Score", float(f1_score))]
    metrics_df = spark.createDataFrame(metrics_data, ["Metric", "Value"])

    # Show the metrics
    print("\nMetrics table:")
    metrics_df.show()

    os.makedirs("output", exist_ok=True)
    
    # Write metrics to CSV
    metrics_df.write.csv("output/task5_output", header=True, mode="overwrite")

    # Confusion matrix (optional but useful)
    print("\nConfusion Matrix:")
    predictions.groupBy("label_index", "prediction").count().show()

    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()