package com.hiomiai.resultsservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class CallbackPayload {
  @NotBlank
  private String filename;
  @NotBlank private String userId;
  @NotBlank private String contentType;
  @NotNull
  private Long size;
  @NotNull private Result result;

  @Data
  public static class Result {
    @NotBlank private String prediction; // "Normal" | "Pneumonia"
    @NotNull private Double confidence;
    @NotNull private Probabilities probabilities;
  }

  @Data
  public static class Probabilities {
    @NotNull private Double normal;
    @NotNull private Double pneumonia;
  }

}
