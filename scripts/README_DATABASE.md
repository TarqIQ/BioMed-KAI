# Database Setup Guide

This guide explains how to set up the PostgreSQL database for BioMed-KAI chat storage.

## Prerequisites

- PostgreSQL installed and running
- Python 3.8+ with required packages (sqlalchemy, psycopg2)
- Environment variables configured (see below)

## Environment Variables

Make sure the following environment variables are set in your `.env` file or environment:

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=biomedkai_chats
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
```

## Setup Steps

1. **Ensure PostgreSQL is running:**
   ```bash
   # Check if PostgreSQL is running
   pg_isready
   ```

2. **Run the setup script:**
   ```bash
   python scripts/setup_database.py
   ```

   Or make it executable and run directly:
   ```bash
   chmod +x scripts/setup_database.py
   ./scripts/setup_database.py
   ```

3. **Verify the setup:**
   ```bash
   # Connect to PostgreSQL
   psql -U your_username -d biomedkai_chats
   
   # Check tables
   \dt
   
   # You should see:
   # - chats
   # - messages
   # - chat_metadata
   ```

## Database Schema

### Tables Created

1. **chats** - Stores chat sessions
   - `id` (VARCHAR, PRIMARY KEY) - Chat session ID
   - `user_id` (VARCHAR) - User identifier
   - `title` (TEXT) - Chat title
   - `created_at` (TIMESTAMP) - Creation timestamp
   - `updated_at` (TIMESTAMP) - Last update timestamp
   - `metadata` (JSONB) - Additional metadata

2. **messages** - Stores individual messages
   - `id` (SERIAL, PRIMARY KEY) - Message ID
   - `chat_id` (VARCHAR, FOREIGN KEY) - Reference to chat
   - `role` (VARCHAR) - Message role (user/assistant)
   - `content` (TEXT) - Message content
   - `metadata` (JSONB) - Message metadata
   - `created_at` (TIMESTAMP) - Creation timestamp

3. **chat_metadata** - Stores insights and metadata
   - `id` (SERIAL, PRIMARY KEY) - Metadata ID
   - `chat_id` (VARCHAR, FOREIGN KEY) - Reference to chat
   - `message_id` (INTEGER, FOREIGN KEY) - Reference to message
   - `kg_data` (JSONB) - Knowledge graph data
   - `references_data` (JSONB) - References/publications
   - `agent_selection` (JSONB) - Agent selection criteria
   - `created_at` (TIMESTAMP) - Creation timestamp

### Indexes

The setup script creates indexes for optimal query performance:
- `idx_chats_user_id` - For user-based queries
- `idx_chats_created_at` - For time-based sorting
- `idx_messages_chat_id` - For message retrieval by chat
- `idx_messages_created_at` - For chronological ordering
- `idx_chat_metadata_chat_id` - For metadata retrieval
- `idx_chat_metadata_message_id` - For message-specific metadata

## Troubleshooting

### Database Connection Errors

If you encounter connection errors:
1. Verify PostgreSQL is running: `pg_isready`
2. Check credentials in environment variables
3. Ensure the database user has CREATE DATABASE privileges
4. Check firewall/network settings if connecting remotely

### Permission Errors

If you get permission errors:
```sql
-- Grant necessary privileges (run as postgres superuser)
GRANT ALL PRIVILEGES ON DATABASE biomedkai_chats TO your_username;
```

### Table Already Exists

The script uses `CREATE TABLE IF NOT EXISTS`, so it's safe to run multiple times. It will skip existing tables.

## Manual Setup (Alternative)

If you prefer to set up manually:

```sql
-- Connect as postgres superuser
psql -U postgres

-- Create database
CREATE DATABASE biomedkai_chats;

-- Connect to the new database
\c biomedkai_chats

-- Run the SQL from setup_database.py manually
```

## Usage

Once set up, the chat storage is automatically used by the BioMed-KAI backend. All conversations are persisted to the database, including:
- Chat sessions
- User and assistant messages
- Knowledge graph data
- References
- Agent selection metadata

## Backup and Maintenance

### Backup
```bash
pg_dump -U your_username biomedkai_chats > backup.sql
```

### Restore
```bash
psql -U your_username biomedkai_chats < backup.sql
```

### Vacuum (for maintenance)
```sql
VACUUM ANALYZE;
```

