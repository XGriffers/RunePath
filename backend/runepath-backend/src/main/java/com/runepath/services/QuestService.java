package com.runepath.services;

import com.runepath.models.Quest;

import java.util.List;

public interface QuestService {
    List<Quest> getAllQuests();
    Quest createQuest(Quest quest);
}
