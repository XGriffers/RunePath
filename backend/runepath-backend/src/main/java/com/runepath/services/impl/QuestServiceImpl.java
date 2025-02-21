package com.runepath.services.impl;

import com.runepath.models.Quest;
import com.runepath.repositories.QuestRepository;
import com.runepath.services.QuestService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class QuestServiceImpl implements QuestService {

    @Autowired
    private QuestRepository questRepository;

    @Override
    public List<Quest> getAllQuests() {
        return questRepository.findAll();
    }

    @Override
    public Quest createQuest(Quest quest) {
        return questRepository.save(quest);
    }
}
