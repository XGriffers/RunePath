package com.runepath.controllers;

import com.runepath.models.Progress;
import com.runepath.services.ProgressService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/progress")
public class ProgressController {
    @Autowired
    private ProgressService progressService;

    @GetMapping("/{userId}")
    public List<Progress> getProgressByUserId(@PathVariable Long userId) {
        return progressService.getProgressByUserId(userId);
    }

    @PostMapping
    public Progress createProgress(@RequestBody Progress progress) {
        return progressService.createProgress(progress);
    }
}


