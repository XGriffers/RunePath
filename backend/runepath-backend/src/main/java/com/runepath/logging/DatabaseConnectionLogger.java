package com.runepath.logging;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

@Component
public class DatabaseConnectionLogger {
    private static final Logger logger = LoggerFactory.getLogger(DatabaseConnectionLogger.class);

    private final DataSourceProperties dataSourceProperties;

    public DatabaseConnectionLogger(DataSourceProperties dataSourceProperties) {
        this.dataSourceProperties = dataSourceProperties;
    }

    @EventListener(org.springframework.boot.context.event.ApplicationStartedEvent.class)
    public void logDatabaseConnectionInfo() {
        logger.info("Database connection established");
        logger.info("Database URL: {}", dataSourceProperties.getUrl());
        logger.info("Database Username: {}", dataSourceProperties.getUsername());
    }
}
