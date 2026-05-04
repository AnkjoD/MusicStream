import ast
import sys

files = [
    '/opt/airflow/src/jobs/batch/raw_to_silver.py',
    '/opt/airflow/src/jobs/batch/train_recommendation.py',
    '/opt/airflow/src/jobs/batch/churn_prediction.py',
    '/opt/airflow/src/jobs/streaming/kafka_to_minio.py',
    '/opt/airflow/src/dags/kafka_to_minio_dag.py',
    '/opt/airflow/src/dags/medallion_batch_dag.py',
]

all_ok = True
for f in files:
    name = f.split('/')[-1]
    try:
        content = open(f, encoding='utf-8').read()
        ast.parse(content)
        print(f'✅ OK: {name}')
    except SyntaxError as e:
        print(f'❌ FAIL: {name} -> Line {e.lineno}: {e.msg}')
        all_ok = False

if not all_ok:
    sys.exit(1)
else:
    print('\n✅ All files passed syntax check!')
