package com.hiomiai.upload_minioservice.controller;

import com.hiomiai.upload_minioservice.services.MinioUploadService;
import com.hiomiai.upload_minioservice.ml.MlServiceClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/upload")
@CrossOrigin(origins = "http://localhost:3000") // Add this line
public class UploadController {
  private static final Logger log = LoggerFactory.getLogger(UploadController.class);

  private final MinioUploadService uploadService;
  private final MlServiceClient mlServiceClient;

  public UploadController(MinioUploadService uploadService, MlServiceClient mlServiceClient) {
    this.uploadService = uploadService;
    this.mlServiceClient = mlServiceClient;
  }

  @PostMapping
  public ResponseEntity<String> upload(
      @RequestParam("file") MultipartFile file
  // @RequestParam("userId") String userId
  ) throws IOException {
    String userId = "1";
    String contentType = file.getContentType();
    String fileName = file.getOriginalFilename();
    long size = file.getSize();

    log.info("Uploading file: {} (size: {}, type: {})", fileName, size, contentType);

    uploadService.uploadFile(fileName, file.getBytes(), contentType);
    log.info("File uploaded successfully to MinIO: {}", fileName);

    try {
      log.info("Notifying ML service for file: {}", fileName);
      mlServiceClient.notifyUpload(fileName, userId, contentType, size);
      log.info("ML service notification sent successfully for file: {}", fileName);
    } catch (Exception e) {
      // Don't fail the upload if ML notification fails; just log it
      log.error("ML notify failed for file '{}' (userId={}): {}", fileName, userId, e.getMessage(), e);
    }

    return ResponseEntity.ok("Uploaded");
  }
}
