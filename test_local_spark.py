import os
import sys

os.environ["JAVA_HOME"] = r"C:\Users\Ankkun\miniconda3\envs\aiEnv\Library"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
# Bỏ qua truy xuất EC2 metadata để tránh treo 2 phút
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
os.environ["AWS_REGION"] = "us-east-1"

from pyspark.sql import SparkSession

jar1 = "file:///C:/Users/Ankkun/Documents/lap_trinh/my_project/streamlify/jars/hadoop-aws-3.3.4.jar"
jar2 = "file:///C:/Users/Ankkun/Documents/lap_trinh/my_project/streamlify/jars/aws-java-sdk-bundle-1.12.262.jar"

spark = SparkSession.builder.appName("Local_Notebook") \
    .config("spark.jars", f"{jar1},{jar2}") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://127.0.0.1:9090") \
    .config("spark.hadoop.fs.s3a.access.key", "homura_madoka") \
    .config("spark.hadoop.fs.s3a.secret.key", "homura123") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .config("spark.hadoop.fs.s3a.endpoint.region", "us-east-1") \
    .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
    .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000") \
    .config("spark.hadoop.fs.s3a.socket.timeout", "60000") \
    .getOrCreate()

print("Spark initialized!", flush=True)

try:
    print("Trying to read one specific file...", flush=True)
    df_single = spark.read.parquet("s3a://bronze-zone/datalake/raw/eventsim/page=About/part-00000-0715900e-343f-42fd-b402-3ab84679caea.c000.snappy.parquet")
    df_single.show(1)
    print("Read ONE file successfully!", flush=True)
except Exception as e:
    print(f"Failed to read one file: {e}", flush=True)

try:
    print("Trying to read the whole directory...", flush=True)
    df_all = spark.read.parquet("s3a://bronze-zone/datalake/raw/eventsim")
    df_all.show(1)
    print("Read ALL files successfully!", flush=True)
except Exception as e:
    print(f"Failed to read all files: {e}", flush=True)
