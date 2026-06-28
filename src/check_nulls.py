from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, when

spark = SparkSession.builder.appName("DebugNullCheck") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "homura_madoka") \
    .config("spark.hadoop.fs.s3a.secret.key", "homura123") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

def check_nulls(path, label):
    print(f"\n=== {label}: {path} ===")
    try:
        df = spark.read.parquet(path)
        total = df.count()
        print(f"Tổng số dòng: {total}")
        null_counts = df.select([
            spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in df.columns
        ])
        null_counts.show(truncate=False, vertical=True)
    except Exception as e:
        print(f"Lỗi khi đọc {label}: {e}")

check_nulls("s3a://bronze-zone/datalake/raw/eventsim", "BRONZE")
check_nulls("s3a://silver-zone/datalake/silver/eventsim", "SILVER")
