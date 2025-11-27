#!/usr/bin/env python3
"""
Database setup script for BioMed-KAI chat storage
Creates necessary tables for storing chat conversations and metadata
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from biomedkai_api.config.settings import settings
import structlog

logger = structlog.get_logger()

def create_database():
    """Create database if it doesn't exist"""
    # Connect to postgres database to create the target database
    admin_url = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/postgres"
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    
    try:
        with admin_engine.connect() as conn:
            # Check if database exists
            result = conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{settings.postgres_db}'")
            )
            exists = result.fetchone()
            
            if not exists:
                logger.info(f"Creating database: {settings.postgres_db}")
                conn.execute(text(f'CREATE DATABASE "{settings.postgres_db}"'))
                logger.info(f"Database {settings.postgres_db} created successfully")
            else:
                logger.info(f"Database {settings.postgres_db} already exists")
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise
    finally:
        admin_engine.dispose()

def create_tables():
    """Create all necessary tables"""
    database_url = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as conn:
            # Create chats table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chats (
                    id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255),
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """))
            
            # Create messages table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    chat_id VARCHAR(255) NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create indexes for messages table
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
                CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
            """))
            
            # Create chat_metadata table for storing insights
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_metadata (
                    id SERIAL PRIMARY KEY,
                    chat_id VARCHAR(255) NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
                    kg_data JSONB,
                    references_data JSONB,
                    agent_selection JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create indexes for chat_metadata table
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chat_metadata_chat_id ON chat_metadata(chat_id);
                CREATE INDEX IF NOT EXISTS idx_chat_metadata_message_id ON chat_metadata(message_id);
            """))
            
            # Additional indexes for better query performance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);
                CREATE INDEX IF NOT EXISTS idx_chats_created_at ON chats(created_at);
            """))
            
            conn.commit()
            logger.info("All tables created successfully")
            
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise
    finally:
        engine.dispose()

def main():
    """Main setup function"""
    logger.info("Starting database setup...")
    
    try:
        # Create database if it doesn't exist
        create_database()
        
        # Create tables
        create_tables()
        
        logger.info("Database setup completed successfully!")
        print("\n✅ Database setup completed successfully!")
        print(f"   Database: {settings.postgres_db}")
        print(f"   Host: {settings.postgres_host}:{settings.postgres_port}")
        print(f"   User: {settings.postgres_user}")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        print(f"\n❌ Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

