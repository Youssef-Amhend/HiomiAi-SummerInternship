package com.hiomiai.upload_minioservice.ml;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;

@Slf4j
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

    webClient.post()
        .uri(notifyPath)
        .contentType(MediaType.APPLICATION_JSON)
        .bodyValue(payload)
        .exchangeToMono(response -> handleResponse(response, filename, userId))
        .timeout(Duration.ofSeconds(5))
        .doOnSuccess(v -> log.info("ML service notification acknowledged for file='{}' (userId={})", filename, userId))
        .block(); // let errors propagate so caller can handle/log them
  }

  private Mono<Void> handleResponse(ClientResponse response, String filename, String userId) {
    if (response.statusCode().is2xxSuccessful()) {
      return Mono.empty();
    }
    return response.bodyToMono(String.class)
        .defaultIfEmpty("")
        .flatMap(body -> {
          String msg = String.format(
              "ML notify failed for file='%s' (userId=%s): status=%s body=%s",
              filename, userId, response.statusCode(), body
          );
          log.error(msg);
          return Mono.error(new IllegalStateException(msg));
        });
  }
}
