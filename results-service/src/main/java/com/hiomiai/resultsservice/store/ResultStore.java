package com.hiomiai.resultsservice.store;

import com.hiomiai.resultsservice.dto.CallbackPayload;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class ResultStore {
  private final Map<String, CallbackPayload> data = new ConcurrentHashMap<>();

  public static String key(String userId, String filename) {
    return userId + ":" + filename;
  }

  public void save(CallbackPayload payload) {
    data.put(key(payload.getUserId(), payload.getFilename()), payload);
  }

  public Optional<CallbackPayload> get(String userId, String filename) {
    return Optional.ofNullable(data.get(key(userId, filename)));
  }

}
