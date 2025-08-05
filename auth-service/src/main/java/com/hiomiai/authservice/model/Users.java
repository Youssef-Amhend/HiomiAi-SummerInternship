package com.hiomiai.authservice.model;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

import java.security.Timestamp;
import java.util.UUID;

@Entity
public class Users {
  @Id
  @GeneratedValue(strategy = GenerationType.UUID)
  private UUID id;
  private String username;
  private String password;
  private String email;
//  private Timestamp created;
//  private Timestamp updated;
  private Boolean is_active;
  private String role;

  public String getUsername() {
    return username;
  }

  public void setUsername(String username) {
    this.username = username;
  }

  public String getPassword() {
    return password;
  }

  public void setPassword(String password) {
    this.password = password;
  }

  public String getEmail() {
    return email;
  }

  public void setEmail(String email) {
    this.email = email;
  }

//  public Timestamp getCreated() {
//    return created;
//  }
//
//  public void setCreated(Timestamp created) {
//    this.created = created;
//  }
//
//  public Timestamp getUpdated() {
//    return updated;
//  }
//
//  public void setUpdated(Timestamp updated) {
//    this.updated = updated;
//  }

  public Boolean getIs_active() {
    return is_active;
  }

  public void setIs_active(Boolean is_active) {
    this.is_active = is_active;
  }

  public String getRole() {
    return role;
  }

  public void setRole(String role) {
    this.role = role;
  }


  public void setId(UUID id) {
    this.id = id;
  }

  public UUID getId() {
    return id;
  }
}
