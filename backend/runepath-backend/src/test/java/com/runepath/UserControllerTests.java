package com.runepath;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.mockito.Mock;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import com.runepath.controllers.UserController;
import com.runepath.models.User;
import com.runepath.services.UserService;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

import java.util.Arrays;
import java.util.List;

@WebMvcTest(UserController.class)
class UserControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Mock
    private UserService userService;

    @Test
    void testGetAllUsers() throws Exception {
        List<User> users = Arrays.asList(
            new User("user1", true),
            new User("user2", false)
        );
        when(userService.getAllUsers()).thenReturn(users);

        mockMvc.perform(get("/api/users"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$[0].username").value("user1"))
               .andExpect(jsonPath("$[1].username").value("user2"));
    }

    @Test
    void testCreateUser() throws Exception {
        User newUser = new User("newuser", true);
        when(userService.createUser(any(User.class))).thenReturn(newUser);

        mockMvc.perform(post("/api/users")
               .contentType(MediaType.APPLICATION_JSON)
               .content("{\"username\":\"newuser\",\"isMember\":true}"))
               .andExpect(status().isCreated())
               .andExpect(jsonPath("$.username").value("newuser"));
    }

    @Test
    void testGetUserByUsername() throws Exception {
        User user = new User("testuser", true);
        when(userService.getUserByUsername("testuser")).thenReturn(user);

        mockMvc.perform(get("/api/users/testuser"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.username").value("testuser"));
    }

    @Test
    void testDeleteUser() throws Exception {
        mockMvc.perform(delete("/api/users/1"))
               .andExpect(status().isOk());

        verify(userService, times(1)).deleteUser(1L);
    }
    @Test
void testGetNonExistentUser() throws Exception {
    when(userService.getUserByUsername("nonexistent")).thenReturn(null);

    mockMvc.perform(get("/api/users/nonexistent"))
           .andExpect(status().isNotFound());
}

@Test
void testCreateUserWithInvalidData() throws Exception {
    mockMvc.perform(post("/api/users")
           .contentType(MediaType.APPLICATION_JSON)
           .content("{\"username\":\"\",\"isMember\":true}"))
           .andExpect(status().isBadRequest());
}

@Test
void testUpdateUser() throws Exception {
    User updatedUser = new User("updateduser", false);
    when(userService.updateUser(any(User.class), eq(1L))).thenReturn(updatedUser);

    mockMvc.perform(put("/api/users/1")
           .contentType(MediaType.APPLICATION_JSON)
           .content("{\"username\":\"updateduser\",\"isMember\":false}"))
           .andExpect(status().isOk())
           .andExpect(jsonPath("$.username").value("updateduser"))
           .andExpect(jsonPath("$.isMember").value(false));
}

}
