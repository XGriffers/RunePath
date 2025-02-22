package com.runepath.models;

import java.sql.Timestamp;

import jakarta.persistence.*;

@Entity
@Table(name = "quests")
public class Quest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String title;

    private String requirements;

    private String rewards;

    @Column(nullable = false)
    private Integer difficulty;

    @Column(nullable = false)
    private boolean isMembers;

    @Column(nullable = false)
    private Timestamp created_at;

    public Quest() {}
    
    public Quest(Timestamp created_at, String title, String requirements, String rewards, Integer difficulty, boolean isMembers) {
        this.title = title;
        this.requirements = requirements;
        this.rewards = rewards;
        this.difficulty = difficulty;
        this.isMembers = isMembers;
        this.created_at = created_at;
    }

    // Getters and setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getRequirements() {
        return requirements;
    }

    public void setRequirements(String requirements) {
        this.requirements = requirements;
    }

    public String getRewards() {
        return rewards;
    }

    public void setRewards(String rewards) {
        this.rewards = rewards;
    }

    public Integer getDifficulty() {
        return difficulty;
    }

    public void setDifficulty(Integer difficulty) {
        this.difficulty = difficulty;
    }

    public boolean isMembers() {
        return isMembers;
    }

    public void setMembers(boolean members) {
        isMembers = members;
    }
}
