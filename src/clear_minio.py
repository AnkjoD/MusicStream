import boto3
from botocore.client import Config

s3 = boto3.client(
    's3',
    endpoint_url='http://minio:9000',
    aws_access_key_id='homura_madoka',
    aws_secret_access_key='homura123',
    config=Config(signature_version='s3v4')
)

for bucket in ['bronze-zone', 'silver-zone', 'gold-zone']:
    try:
        paginator = s3.get_paginator('list_objects_v2')
        total = 0
        for page in paginator.paginate(Bucket=bucket):
            contents = page.get('Contents', [])
            if contents:
                delete_keys = [{'Key': obj['Key']} for obj in contents]
                s3.delete_objects(Bucket=bucket, Delete={'Objects': delete_keys})
                total += len(delete_keys)
        print(f'[OK] Cleared {total} objects from {bucket}')
    except Exception as e:
        print(f'[ERR] {bucket}: {e}')
