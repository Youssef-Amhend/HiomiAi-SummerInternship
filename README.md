# HiomiAI Summer Internship Project

A microservices-based application for pneumonia detection using ML, with MinIO storage and FastAPI/Spring Boot services.

## Architecture

- **ML Service**: FastAPI service for pneumonia detection using ResNet18
- **Upload Service**: Spring Boot service for file uploads to MinIO
- **Results Service**: Spring Boot service for storing and streaming ML results
- **MinIO**: Object storage service
- **PostgreSQL**: Database service
- **Redis**: Cache service

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Java 17 (for local development)
- Python 3.9+ (for local development)

### Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd HiomiAi-SummerInternship
   ```

2. **Start all services**

   ```bash
   docker-compose up --build
   ```

3. **Access services**
   - Upload Service: http://localhost:8080
   - ML Service: http://localhost:8000
   - Results Service: http://localhost:8085
   - MinIO Console: http://localhost:9001
   - PostgreSQL: localhost:5432
   - Redis: localhost:6379

### Service Endpoints

#### Upload Service (Port 8080)

- `POST /upload` - Upload an image file

#### ML Service (Port 8000)

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /ml/upload-notify` - Process uploaded image
- `GET /ml/result` - Get prediction result
- `POST /predict` - Direct image prediction
- `POST /admin/reload-model` - Reload ML model

#### Results Service (Port 8085)

- `POST /ml-callback` - Receive ML prediction results
- `GET /results` - Get stored result by userId and filename
- `GET /stream` - Server-Sent Events stream for live updates

### Troubleshooting

#### ML Service Not Responding

1. **Check if ML service is running**

   ```bash
   docker-compose ps
   ```

2. **Check ML service logs**

   ```bash
   docker-compose logs ml-service
   ```

3. **Check if model is loaded**

   ```bash
   curl http://localhost:8000/health
   ```

4. **Reload model if needed**
   ```bash
   curl -X POST http://localhost:8000/admin/reload-model
   ```

#### Upload Service Issues

1. **Check upload service logs**

   ```bash
   docker-compose logs upload-service
   ```

2. **Verify MinIO connection**
   ```bash
   docker-compose logs minio
   ```

#### Network Issues

1. **Check service connectivity**

   ```bash
   docker-compose exec upload-service ping ml-service
   ```

2. **Verify environment variables**
   ```bash
   docker-compose exec ml-service env | grep S3
   ```

### Development

#### Running Services Locally

1. **ML Service**

   ```bash
   cd ml-service
   pip install -r requirements.txt
   python main.py
   ```

2. **Upload Service**
   ```bash
   cd upload_minio-service
   ./mvnw spring-boot:run
   ```

#### Environment Variables

Create `.env` files in respective service directories:

**ML Service (.env)**

```
APP_HOST=0.0.0.0
APP_PORT=8000
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=murmax
S3_SECRET_KEY=rootroot
S3_REGION=us-east-1
S3_BUCKET=s3bucket
S3_USE_SSL=false
MODEL_PATH=resnet18_pneumonia_best.pth
```

### Testing

1. **Test upload and ML processing**

   ```bash
   curl -X POST -F "file=@test-image.jpg" http://localhost:8080/upload
   ```

2. **Check prediction result**

   ```bash
   curl "http://localhost:8000/ml/result?userId=1&filename=test-image.jpg"
   ```

3. **Check results service**

   ```bash
   curl "http://localhost:8085/results?userId=1&filename=test-image.jpg"
   ```

4. **Stream live updates**
   ```bash
   curl "http://localhost:8085/stream"
   ```

### Common Issues and Solutions

1. **Model not loading**: Ensure the model file exists and has correct permissions
2. **MinIO connection failed**: Check MinIO credentials and endpoint URL
3. **Service communication failed**: Verify network configuration in docker-compose
4. **Memory issues**: Increase Docker memory allocation for ML model loading
