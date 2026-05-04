FROM python:3.11-slim

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Docker CLI (để Dashboard điều khiển được các container khác)
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Cài đặt các thư viện Python cho Dashboard
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn dashboard
COPY src/dashboard /app/dashboard

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
