package com.runepath.models;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Valid;
import jakarta.persistence.*;

@Entity
@Table(name = "users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank(message = "Username is required")

    @Column(unique = true, nullable = false)
    private String username;

    @Column(nullable = false)
    @NotNull(message = "Member status is required")
    private boolean isMember;

    // Default constructor
    public User() {}

    // Constructor with fields
    public User(String username, boolean isMember) {
        this.username = username;
        this.isMember = isMember;
    }

    // Getters and setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public boolean isMember() {
        return isMember;
    }

    public void setMember(boolean member) {
        isMember = member;
    }
}
