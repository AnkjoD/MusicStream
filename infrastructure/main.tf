terraform {
  required_providers {
    minio = {
      source  = "aminueza/minio"
      version = "~> 2.0.0"
    }
  }
}
variable "minio_user" {
  description = "Tài khoản MinIO"
  type        = string
  sensitive   = true # Che dữ liệu trên terminal
}

variable "minio_password" {
  description = "Mật khẩu MinIO"
  type        = string
  sensitive   = true # Che dữ liệu trên terminal
}
# Khai báo kết nối đến MinIO (local)
provider "minio" {
  minio_server   = "127.0.0.1:9090"
  minio_user     = var.minio_user
  minio_password = var.minio_password
  minio_ssl      = false
}

# Tầng 1: Bronze (Chứa dữ liệu thô từ Kafka)
resource "minio_s3_bucket" "bronze_zone" {
  bucket = "bronze-zone"
  acl    = "public-read-write"
}

# Tầng 2: Silver (Chứa dữ liệu đã làm sạch, định dạng bảng Iceberg)
resource "minio_s3_bucket" "silver_zone" {
  bucket = "silver-zone"
  acl    = "public-read-write"
}

# Tầng 3: Gold (Chứa dữ liệu tổng hợp cho báo cáo / AI Training)
resource "minio_s3_bucket" "gold_zone" {
  bucket = "gold-zone"
  acl    = "public-read-write"
}