package com.runepath;

import java.sql.Timestamp;
import java.time.Instant;
import java.util.List;
import java.util.Collections;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

import com.runepath.controllers.ProgressController;
import com.runepath.models.Progress;
import com.runepath.models.User;
import com.runepath.models.Quest;
import com.runepath.services.ProgressService;

@WebMvcTest(ProgressController.class)
@ExtendWith(MockitoExtension.class)
class ProgressControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Mock
    private ProgressService progressService;

    @Test
    void testGetProgressForUser() throws Exception {
        User user = new User("testuser", true);
        Quest quest = new Quest(Timestamp.from(Instant.now()), "TestQuest", "TestReq", "TestReward", 2, false, false, null);
        Progress progress = new Progress(Timestamp.from(Instant.now()), user, quest, false, null);
        List<Progress> progressList = List.of(progress);
    
        when(progressService.getProgressForUser(1L)).thenReturn(progressList);
    
        mockMvc.perform(get("/api/progress/user/1"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$[0].quest.title").value("TestQuest"))
               .andExpect(jsonPath("$[0].completed").value(false));
    }
    
    @Test
    void testCreateProgress() throws Exception {
        User user = new User("testuser", true);
        Quest quest = new Quest(Timestamp.from(Instant.now()), "TestQuest", "TestReq", "TestReward", 2, false, false, null);
        Progress newProgress = new Progress(Timestamp.from(Instant.now()), user, quest, false, null);
    
        when(progressService.createProgress(any(Progress.class))).thenReturn(newProgress);
    
        mockMvc.perform(post("/api/progress")
               .contentType(MediaType.APPLICATION_JSON)
               .content("{\"user\":{\"id\":1},\"quest\":{\"id\":1},\"completed\":false}"))
               .andExpect(status().isCreated())
               .andExpect(jsonPath("$.quest.title").value("TestQuest"));
    }
    
    @Test
    void testUpdateProgress() throws Exception {
        User user = new User("testuser", true);
        Quest quest = new Quest(Timestamp.from(Instant.now()), "TestQuest", "TestReq", "TestReward", 2, false, false, null);
        Progress updateProgress = new Progress(Timestamp.from(Instant.now()), user, quest, true, Timestamp.from(Instant.now()));
    
        when(progressService.updateProgress(eq(1L), any(Progress.class))).thenReturn(updateProgress);
    
        mockMvc.perform(put("/api/progress/1")
               .contentType(MediaType.APPLICATION_JSON)
               .content("{\"user\":{\"id\":1},\"quest\":{\"id\":1},\"completed\":true}"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.completed").value(true));
    }
    
    @Test
void testGetProgressForUserWithNoProgress() throws Exception {
    when(progressService.getProgressForUser(2L)).thenReturn(Collections.emptyList());

    mockMvc.perform(get("/api/progress/user/2"))
           .andExpect(status().isOk())
           .andExpect(jsonPath("$").isArray())
           .andExpect(jsonPath("$").isEmpty());
}


@Test
void testDeleteProgress() throws Exception {
    mockMvc.perform(delete("/api/progress/1"))
           .andExpect(status().isOk());

    verify(progressService, times(1)).deleteProgress(1L);
}

}
