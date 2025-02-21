package com.runepath.services.impl;

import com.runepath.models.Progress;
import com.runepath.repositories.ProgressRepository;
import com.runepath.services.ProgressService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ProgressServiceImpl implements ProgressService {

    @Autowired
    private ProgressRepository progressRepository;

    @Override
    public List<Progress> getProgressByUserId(Long userId) {
        // Implement logic to retrieve progress by user ID if needed.
        return null; // Placeholder for actual implementation.
    }

    @Override
    public Progress createProgress(Progress progress) {
        return progressRepository.save(progress);
    }
}
