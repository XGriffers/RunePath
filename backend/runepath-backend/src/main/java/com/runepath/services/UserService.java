package com.runepath.services;

import com.runepath.models.User;

import java.util.List;

public interface UserService {
    User getUserByUsername(String username);
    List<User> getAllUsers();
    User createUser(User user);
    void deleteUser(Long id);
}
