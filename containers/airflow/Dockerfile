# ============================================================
# Custom Airflow Image - Baked-in providers + spark-submit binary + JARs
# ============================================================
ARG AIRFLOW_VERSION=2.10.5
ARG SPARK_VERSION=3.5.4

# Stage 1: Lấy spark base
FROM apache/spark:${SPARK_VERSION} AS spark-base

# Stage 2: Build Airflow image chính
FROM apache/airflow:${AIRFLOW_VERSION}-python3.11

USER root

# Copy Spark runtime
COPY --from=spark-base /opt/spark /opt/spark

# Cài Java
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set envs
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"

# Thêm JARs cần thiết trực tiếp vào Airflow worker (vì driver chạy ở đây)
ENV HADOOP_AWS_VERSION=3.3.4
ENV AWS_SDK_VERSION=1.12.262
ENV KAFKA_SPARK_VERSION=3.5.4
ENV KAFKA_CLIENT_VERSION=3.4.1

ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/${KAFKA_SPARK_VERSION}/spark-sql-kafka-0-10_2.12-${KAFKA_SPARK_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/${KAFKA_CLIENT_VERSION}/kafka-clients-${KAFKA_CLIENT_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/${KAFKA_SPARK_VERSION}/spark-token-provider-kafka-0-10_2.12-${KAFKA_SPARK_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar /opt/spark/jars/

RUN chmod 644 /opt/spark/jars/*.jar

USER airflow

# Cài providers
RUN pip install --no-cache-dir --no-deps \
    apache-airflow-providers-apache-spark==4.10.0

RUN pip install --no-cache-dir \
    grpcio-status>=1.59.0

RUN pip install --no-cache-dir \
    apache-airflow-providers-apache-kafka==1.4.0 \
    minio==7.2.15 \
    boto3==1.35.99
