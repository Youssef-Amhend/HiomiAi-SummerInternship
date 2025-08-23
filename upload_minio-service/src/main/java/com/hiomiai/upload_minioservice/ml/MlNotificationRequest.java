package com.hiomiai.upload_minioservice.ml;

public record MlNotificationRequest(String filename,
        String userId,
        String contentType,
        long size,
        String callbackUrl) {
}
