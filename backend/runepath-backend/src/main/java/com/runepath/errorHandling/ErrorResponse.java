package com.runepath.errorHandling;

public class ErrorResponse {
    private String code;
    private String message;
    
        public String getCode() {
            return this.code;
        }
    

        public void setCode(String code) {
            this.code = code;
        }
    
        
        public String getMessage() {
            return this.message;
        }
    
        
        public void setMessage(String message) {
            this.message = message;
        }

    public ErrorResponse(String code, String message) {
        this.code = code;
        this.message = message;
    }
    
}