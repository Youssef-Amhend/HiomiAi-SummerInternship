package com.hiomiai.upload_minioservice.services;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

@Service
public class MinioUploadService {
  @Value("${minio.bucket}")
  private String bucket;

  private final S3Client s3Client;

  public MinioUploadService(S3Client s3Client) {
    this.s3Client = s3Client;
  }

  public void uploadFile(String filename, byte[] content , String contentType) {

    s3Client.putObject(
        PutObjectRequest.builder()
            .bucket(bucket)
            .key(filename)
            .contentType(contentType)
            .build(),
        RequestBody.fromBytes(content)
    );
  }
}
