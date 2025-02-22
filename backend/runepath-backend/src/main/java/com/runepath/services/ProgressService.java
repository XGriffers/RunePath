package com.runepath.services;

import com.runepath.models.Progress;
import java.util.List;

public interface ProgressService {
    List<Progress> getProgressForUser(Long userId);
    Progress createProgress(Progress progress);
    Progress updateProgress(Long progressId, Progress progressDetails);
    void deleteProgress(Long progressId);
    Progress getProgressById(Long progressId);
}

