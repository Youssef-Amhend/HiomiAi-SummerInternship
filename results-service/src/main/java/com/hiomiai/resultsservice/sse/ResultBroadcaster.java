package com.hiomiai.resultsservice.sse;

import com.hiomiai.resultsservice.dto.CallbackPayload;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArraySet;

@Component
public class ResultBroadcaster {
  private final Set<SseEmitter> emitters = new CopyOnWriteArraySet<>();

  public SseEmitter register() {
    SseEmitter emitter = new SseEmitter(0L); // no timeout
    emitters.add(emitter);
    emitter.onCompletion(() -> emitters.remove(emitter));
    emitter.onTimeout(() -> emitters.remove(emitter));
    emitter.onError(e -> emitters.remove(emitter));
    return emitter;
  }

  public void broadcast(CallbackPayload payload) {
    emitters.forEach(emitter -> {
      try {
        emitter.send(SseEmitter.event()
            .name("ml-result")
            .data(payload));
      } catch (IOException e) {
        emitter.complete();
        emitters.remove(emitter);
      }
    });
  }

}
