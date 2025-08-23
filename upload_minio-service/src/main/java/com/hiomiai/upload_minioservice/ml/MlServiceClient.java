package com.hiomiai.upload_minioservice.ml;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Component
public class MlServiceClient {

  private final WebClient webClient;
  private final String notifyPath;

  public MlServiceClient(
      @Value("${ml-service.base-url}") String baseUrl,
      @Value("${ml-service.notify-path:/ml/upload-notify}") String notifyPath) {
    this.webClient = WebClient.builder()
        .baseUrl(baseUrl)
        .build();
    this.notifyPath = notifyPath;
  }

  public void notifyUpload(String filename, String userId, String contentType, long size) {
    String callbackUrl = "http://results-service:8085/ml-callback";
    MlNotificationRequest payload = new MlNotificationRequest(filename, userId, contentType, size, callbackUrl);

    // Fire the request; block to ensure it’s sent, but handle errors upstream
    webClient.post()
        .uri(notifyPath)
        .contentType(MediaType.APPLICATION_JSON)
        .bodyValue(payload)
        .retrieve()
        .bodyToMono(Void.class)
        .onErrorResume(ex -> {
          System.err.println("Error notifying ML service: " + ex.getMessage());
          return Mono.empty();
        })
        .block();
  }
}
