# ============================================================
# Spark Dockerfile - Optimized, Pre-baked JARs
# Context: streamlify/ (root)
# ============================================================

FROM apache/spark:3.5.4

USER root

ENV HADOOP_AWS_VERSION=3.3.4
ENV AWS_SDK_VERSION=1.12.262
ENV KAFKA_SPARK_VERSION=3.5.4
ENV KAFKA_CLIENT_VERSION=3.4.1

# ── Pre-bake tất cả JARs (không download lúc runtime) ──
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/${KAFKA_SPARK_VERSION}/spark-sql-kafka-0-10_2.12-${KAFKA_SPARK_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/${KAFKA_CLIENT_VERSION}/kafka-clients-${KAFKA_CLIENT_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/${KAFKA_SPARK_VERSION}/spark-token-provider-kafka-0-10_2.12-${KAFKA_SPARK_VERSION}.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar /opt/spark/jars/

RUN chmod 644 /opt/spark/jars/*.jar && \
    mkdir -p /tmp/.ivy2 && chmod 777 /tmp/.ivy2 && \
    mkdir -p /opt/spark/work-dir && chmod 777 /opt/spark/work-dir

# Context là root (streamlify/), nên path đúng là containers/spark/start-master.sh
COPY containers/spark/start-master.sh /opt/spark/start-master.sh
RUN chmod +x /opt/spark/start-master.sh

USER spark
