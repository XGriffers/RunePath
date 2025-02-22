package com.runepath.services.impl;

import com.runepath.models.Progress;
import com.runepath.repositories.ProgressRepository;
import com.runepath.services.ProgressService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class ProgressServiceImpl implements ProgressService {

    @Autowired
    private ProgressRepository progressRepository;

    @Override
    public List<Progress> getProgressForUser(Long userId) {
        return progressRepository.findByUserId(userId);
    }

    @Override
    public Progress createProgress(Progress progress) {
        return progressRepository.save(progress);
    }

    @Override
    public Progress updateProgress(Long progressId, Progress progressDetails) {
        Optional<Progress> progress = progressRepository.findById(progressId);
        if (progress.isPresent()) {
            Progress existingProgress = progress.get();
            existingProgress.setCompleted(progressDetails.isCompleted());
            existingProgress.setUpdatedProgress(progressDetails.getUpdatedProgress());
            return progressRepository.save(existingProgress);
        }
        return null; // Or throw an exception if the progress doesn't exist
    }

    @Override
    public void deleteProgress(Long progressId) {
        progressRepository.deleteById(progressId);
    }

    @Override
    public Progress getProgressById(Long progressId) {
        return progressRepository.findById(progressId).orElse(null);
    }
}
