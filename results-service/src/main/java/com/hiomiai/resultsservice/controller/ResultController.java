package com.hiomiai.resultsservice.controller;

import com.hiomiai.resultsservice.dto.CallbackPayload;
import com.hiomiai.resultsservice.sse.ResultBroadcaster;
import com.hiomiai.resultsservice.store.ResultStore;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Map;

@RestController
@RequiredArgsConstructor
public class ResultController {
  private static final Logger log = LoggerFactory.getLogger(ResultController.class);

  private final ResultStore store;
  private final ResultBroadcaster broadcaster;

  // Called by ML service when it finishes
  @PostMapping(value = "/ml-callback", consumes = MediaType.APPLICATION_JSON_VALUE)
  public Map<String, Object> receiveCallback(@Valid @RequestBody CallbackPayload payload) {
    // Log
    log.info("ML result: userId={}, filename={}, prediction={}, confidence={}, probs={{normal:{}, pneumonia:{}}}",
        payload.getUserId(),
        payload.getFilename(),
        payload.getResult().getPrediction(),
        payload.getResult().getConfidence(),
        payload.getResult().getProbabilities().getNormal(),
        payload.getResult().getProbabilities().getPneumonia()
    );

    // Store and broadcast to SSE listeners
    store.save(payload);
    broadcaster.broadcast(payload);

    return Map.of(
        "status", "accepted",
        "key", ResultStore.key(payload.getUserId(), payload.getFilename())
    );
  }

  // Your frontend can poll this if not using SSE
  @GetMapping("/results")
  public Object getResult(@RequestParam String userId, @RequestParam String filename) {
    return store.get(userId, filename)
        .<Object>map(r -> r)
        .orElseGet(() -> Map.of("status", "not_found"));
  }

  // Live updates to frontend
  @GetMapping(path = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
  public SseEmitter stream() {
    return broadcaster.register();
  }

}
