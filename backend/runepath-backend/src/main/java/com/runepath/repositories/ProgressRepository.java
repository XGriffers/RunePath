package com.runepath.repositories;

import com.runepath.models.Progress;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ProgressRepository extends JpaRepository<Progress, Long> {
    List<Progress> findByUserId(Long userId);
}