from pyspark.sql import SparkSession

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Fake News Analysis - Task 1") \
        .getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("ERROR")

    print("======= TASK 1: LOAD & BASIC EXPLORATION =======")

    # Read the CSV file, inferring the schema
    df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

    # Create a temporary view
    df.createOrReplaceTempView("news_data")

    # Show the first 5 rows
    print("First 5 rows:")
    df.show(5, truncate=False)

    # Count the total number of articles
    total_count = df.count()
    print(f"Total number of articles: {total_count}")

    # Retrieve the distinct labels
    distinct_labels = df.select("label").distinct().collect()
    print("Distinct labels:")
    for row in distinct_labels:
        print(row["label"])

    # Query using SQL
    print("\nUsing SQL to query the first 5 rows:")
    result = spark.sql("SELECT * FROM news_data LIMIT 5")
    result.show(truncate=False)

    # Write DataFrame to CSV
    df.write.csv("output/task1_output", header=True, mode="overwrite")

    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()