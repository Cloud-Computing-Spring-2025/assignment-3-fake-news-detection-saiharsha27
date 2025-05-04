import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Fake News Analysis - Task 3") \
        .getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("ERROR")

    print("======= TASK 3: FEATURE EXTRACTION =======")

    # Read the CSV file
    df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

    # Text preprocessing (from Task 2)
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    tokenized_df = tokenizer.transform(df)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    cleaned_df = remover.transform(tokenized_df)

    # TF-IDF Vectorization
    # HashingTF: Term Frequency
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
    featurized_df = hashingTF.transform(cleaned_df)

    # IDF: Inverse Document Frequency
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idf_model = idf.fit(featurized_df)
    tfidf_df = idf_model.transform(featurized_df)

    # Label Indexing: Convert string labels to numeric values
    label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
    label_model = label_indexer.fit(tfidf_df)
    indexed_df = label_model.transform(tfidf_df)

    # Show sample of feature extraction
    print("Sample of data after feature extraction:")
    indexed_df.select("id", "filtered_words", "features", "label_index").show(5, truncate=False)

    # Handle complex types for CSV output
    from pyspark.sql.functions import array_join, udf
    from pyspark.sql.types import StringType
    
    # Create a UDF to convert feature vectors to string
    @udf(returnType=StringType())
    def vector_to_string(vector):
        if vector:
            return str(vector)
        else:
            return None
    
    # Convert arrays and vectors to string representations
    features_df = indexed_df.select(
        "id", 
        array_join("filtered_words", " ").alias("filtered_words_str"),
        vector_to_string("features").alias("features_str"),
        "label_index"
    )
    
    # Show sample of data ready for CSV export
    print("Sample of data prepared for CSV export:")
    features_df.show(5, truncate=False)
    
    os.makedirs("output", exist_ok=True)
    
    # Write to CSV
    features_df.write.csv("output/task3_output", header=True, mode="overwrite")

    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()