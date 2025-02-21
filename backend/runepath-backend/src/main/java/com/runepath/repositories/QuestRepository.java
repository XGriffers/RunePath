package com.runepath.repositories;

import com.runepath.models.Quest;
import org.springframework.data.jpa.repository.JpaRepository;

public interface QuestRepository extends JpaRepository<Quest, Long> {
    //space for custom queries
}
