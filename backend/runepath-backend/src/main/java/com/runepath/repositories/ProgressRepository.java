package com.runepath.repositories;

import com.runepath.models.Progress;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ProgressRepository extends JpaRepository<Progress, Long> {
    //space for custom queries
}
