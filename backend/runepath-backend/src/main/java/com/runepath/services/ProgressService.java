package com.runepath.services;

import com.runepath.models.Progress;

import java.util.List;

public interface ProgressService {
    List<Progress> getProgressByUserId(Long userId);
    Progress createProgress(Progress progress);
}
