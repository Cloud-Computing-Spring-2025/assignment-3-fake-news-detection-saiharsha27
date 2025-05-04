import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Fake News Analysis - Task 2") \
        .getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("ERROR")

    print("======= TASK 2: TEXT PREPROCESSING =======")

    # Read the CSV file
    df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

    # Convert all text to lowercase
    # Note: Spark's Tokenizer automatically converts to lowercase by default
    
    # Use Tokenizer to split text into words
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    tokenized_df = tokenizer.transform(df)

    # Remove stopwords using StopWordsRemover
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    cleaned_df = remover.transform(tokenized_df)

    # Create a temporary view for the cleaned data (optional)
    cleaned_df.createOrReplaceTempView("cleaned_news")

    # Select relevant columns
    cleaned_result = cleaned_df.select("id", "title", "filtered_words", "label")

    # Show sample of cleaned data
    print("Sample of cleaned and tokenized data:")
    cleaned_result.show(5, truncate=False)

    # Convert array column to string format before writing to CSV
    from pyspark.sql.functions import array_join
    
    # Convert array to string by joining array elements with spaces
    cleaned_result_for_csv = cleaned_result.withColumn(
        "filtered_words_str", 
        array_join("filtered_words", " ")
    ).drop("filtered_words").withColumnRenamed("filtered_words_str", "filtered_words")
    
    os.makedirs("output", exist_ok=True)
    
    # Write the tokenized output to CSV
    cleaned_result_for_csv.write.csv("output/task2_output", header=True, mode="overwrite")

    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()