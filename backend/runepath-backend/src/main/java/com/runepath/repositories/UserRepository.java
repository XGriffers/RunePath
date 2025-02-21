package com.runepath.repositories;

import com.runepath.models.User;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<User, Long> {
    User findByUsername(String username);
    //space for custom queries
}
