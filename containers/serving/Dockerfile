FROM python:3.11-slim

WORKDIR /app


RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    pyarrow \
    minio==7.2.7 \
    mlflow==2.13.0

COPY src/jobs/dl/serving/api.py /app/api.py
COPY src/jobs/dl/models/train_ncf.py /app/ncf_model.py
COPY src/jobs/dl/models/train_lstm_ae.py /app/ae_model.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]