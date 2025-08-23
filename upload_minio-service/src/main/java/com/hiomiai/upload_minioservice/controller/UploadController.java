package com.hiomiai.upload_minioservice.controller;

import com.hiomiai.upload_minioservice.services.MinioUploadService;
import com.hiomiai.upload_minioservice.services.UploadEventProducer;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/upload")
public class UploadController {
  private final MinioUploadService uploadService;
  private final UploadEventProducer producer;

  public UploadController(MinioUploadService uploadService) {
    this.uploadService = uploadService;
    this.producer = new UploadEventProducer();
  }

  @PostMapping
  public ResponseEntity<String> upload(@RequestParam("file") MultipartFile file) throws IOException {
    String fileContent = file.getContentType();
    String userId = "1";
    producer.publishUploadCreatedEvent(userId , file.getOriginalFilename());
    uploadService.uploadFile(file.getOriginalFilename(), file.getBytes() , fileContent);
    return ResponseEntity.ok("Uploaded");
  }
}
