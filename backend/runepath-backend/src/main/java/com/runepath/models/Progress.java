package com.runepath.models;

import java.sql.Timestamp;

import jakarta.persistence.*;

@Entity
@Table(name = "progress")
public class Progress {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne
    @JoinColumn(name = "quest_id", nullable = false)
    private Quest quest;

    @Column(nullable = false)
    private boolean completed;

    @Column(nullable = false)
    private Timestamp created_at;

    @Column(nullable = true)
    private Timestamp updatedProgress;

    public Progress() {}

    public void updateProgress(User user, Quest quest, boolean completed, Timestamp updatedProgress) {
        this.user = user;
        this.quest = quest;
        this.completed = completed;
        this.updatedProgress = updatedProgress;
    }
    

    public Progress(Timestamp created_at, User user, Quest quest, boolean completed, Timestamp updatedProgress) {
        this.user = user;
        this.quest = quest;
        this.completed = completed;
        this.created_at = created_at;
    }
    
    public Timestamp getUpdatedProgress() {
        return updatedProgress;
    }
    
    public void setUpdatedProgress(Timestamp updatedProgress) {
        this.updatedProgress = updatedProgress;
    }

    // Getters and setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public Quest getQuest() {
        return quest;
    }

    public void setQuest(Quest quest) {
        this.quest = quest;
    }

    public boolean isCompleted() {
        return completed;
    }

    public void setCompleted(boolean completed) {
        this.completed = completed;
    }
    public Timestamp getCreated_at() {
        return created_at;
    }
}
