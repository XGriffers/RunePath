package com.runepath.services;

import com.runepath.models.Quest;

import java.util.List;

public interface QuestService {
    Quest getQuestById(Long id);
    List<Quest> getAllQuests();
    Quest createQuest(Quest quest);
    Quest updateQuest(Quest quest);
    void deleteQuest(Long id);
}
