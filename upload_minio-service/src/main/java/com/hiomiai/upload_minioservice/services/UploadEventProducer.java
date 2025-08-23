package com.hiomiai.upload_minioservice.services;

import com.example.hiomiai.protobuf.UploadEvent;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.springframework.stereotype.Service;

import java.util.Properties;
import java.util.UUID;

@Service
public class UploadEventProducer {
  private final KafkaProducer<String , byte[]> producer;
  private final String TOPIC_NAME = "upload-event";

  public UploadEventProducer(){
    Properties props = new Properties();
    props.put("bootstrap.servers", "broker:9092"); // Kafka broker in Docker Compose
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
    props.put("value.serializer", "org.apache.kafka.common.serialization.ByteArraySerializer");
    this.producer = new KafkaProducer<>(props);
  }

  public void publishUploadCreatedEvent(String userId , String filename) {
    String UploadId = UUID.randomUUID().toString();
    UploadEvent.UploadCreatedEvent uploadCreatedEvent = UploadEvent.UploadCreatedEvent.newBuilder()
        .setUploadId(UploadId)
        .setImageId(filename)
        .setUserId(userId)
        .build();
    ProducerRecord<String , byte[]> record = new ProducerRecord<>(TOPIC_NAME ,
        uploadCreatedEvent.getUploadId(),
        uploadCreatedEvent.toByteArray()
    );
    producer.send(record , (metadata , exception) -> {
      if (exception == null) {
        System.out.printf("Produced record to topic %s partition %d @ offset %d%n",
            metadata.topic(), metadata.partition(), metadata.offset());
      } else {
        System.err.println("Error producing record: " + exception.getMessage());
      }
    });
  }
  public void close() {
    producer.close();
  }
}
