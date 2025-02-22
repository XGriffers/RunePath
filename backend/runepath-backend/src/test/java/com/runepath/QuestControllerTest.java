package com.runepath;

import java.sql.Timestamp;
import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.mockito.Mock;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

import com.runepath.controllers.QuestController;
import com.runepath.models.Quest;
import com.runepath.services.QuestService;

@WebMvcTest(QuestController.class)
class QuestControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Mock
    private QuestService questService;

    @Test
    void testGetAllQuests() throws Exception {
        List<Quest> quests = List.of(
            new Quest(Timestamp.from(Instant.now()), "Quest1", "Req1", "Reward1", 1, true, false, null),
            new Quest(Timestamp.from(Instant.now()), "Quest2", "Req2", "Reward2", 2, false, false, null)
        );

        when(questService.getAllQuests()).thenReturn(quests);

        mockMvc.perform(get("/api/quests"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$[0].title").value("Quest1"))
               .andExpect(jsonPath("$[1].title").value("Quest2"));
    }

    @Test
    void testCreateQuest() throws Exception {
        Quest newQuest = new Quest(Timestamp.from(Instant.now()), "NewQuest", "NewReq", "NewReward", 3, true, false, null);
        when(questService.createQuest(any(Quest.class))).thenReturn(newQuest);

        mockMvc.perform(post("/api/quests")
               .contentType(MediaType.APPLICATION_JSON)
               .content("{\"title\":\"NewQuest\",\"requirements\":\"NewReq\",\"rewards\":\"NewReward\",\"difficulty\":3,\"isMembers\":true}"))
               .andExpect(status().isCreated())
               .andExpect(jsonPath("$.title").value("NewQuest"));
    }

    @Test
    void testGetQuestById() throws Exception {
        Quest quest = new Quest(Timestamp.from(Instant.now()), "TestQuest", "TestReq", "TestReward", 2, false, false, null);
        when(questService.getQuestById(1L)).thenReturn(quest);

        mockMvc.perform(get("/api/quests/1"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.title").value("TestQuest"));
    }

    @Test
void testGetNonExistentQuest() throws Exception {
    when(questService.getQuestById(999L)).thenReturn(null);

    mockMvc.perform(get("/api/quests/999"))
           .andExpect(status().isNotFound());
}

@Test
void testUpdateQuest() throws Exception {
    Quest updatedQuest = new Quest(Timestamp.from(Instant.now()), "UpdatedQuest", "UpdatedReq", "UpdatedReward", 3, true, false, null);
    when(questService.updateQuest(any(Quest.class))).thenReturn(updatedQuest);

    mockMvc.perform(put("/api/quests/1")
           .contentType(MediaType.APPLICATION_JSON)
           .content("{\"title\":\"UpdatedQuest\",\"requirements\":\"UpdatedReq\",\"rewards\":\"UpdatedReward\",\"difficulty\":3,\"isMembers\":true}"))
           .andExpect(status().isOk())
           .andExpect(jsonPath("$.title").value("UpdatedQuest"));
}

@Test
void testDeleteQuest() throws Exception {
    mockMvc.perform(delete("/api/quests/1"))
           .andExpect(status().isOk());

    verify(questService, times(1)).deleteQuest(1L);
}

}
