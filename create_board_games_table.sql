-- Create table for storing 8000 board game embeddings
-- This table is designed to work with your custom Neon database component

CREATE TABLE IF NOT EXISTS board_games_embeddings (
    id SERIAL PRIMARY KEY,
    bgg_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(500) NOT NULL,
    year_published INTEGER,
    min_players INTEGER,
    max_players INTEGER,
    play_time INTEGER,
    min_age INTEGER,
    users_rated INTEGER,
    rating_average DECIMAL(4,2),
    bgg_rank INTEGER,
    complexity_average DECIMAL(3,2),
    owned_users INTEGER,
    mechanics TEXT,
    domains TEXT,
    url TEXT,
    description TEXT,
    -- Vector embedding column for pgvector
    embedding VECTOR(1536), -- Adjust dimension based on your embedding model
    -- Metadata columns
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Indexes for better performance
    CONSTRAINT valid_year CHECK (year_published >= 1800 AND year_published <= 2030),
    CONSTRAINT valid_players CHECK (min_players > 0 AND max_players >= min_players),
    CONSTRAINT valid_play_time CHECK (play_time > 0),
    CONSTRAINT valid_age CHECK (min_age >= 0),
    CONSTRAINT valid_rating CHECK (rating_average >= 0 AND rating_average <= 10),
    CONSTRAINT valid_complexity CHECK (complexity_average >= 0 AND complexity_average <= 5)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_board_games_bgg_id ON board_games_embeddings(bgg_id);
CREATE INDEX IF NOT EXISTS idx_board_games_name ON board_games_embeddings(name);
CREATE INDEX IF NOT EXISTS idx_board_games_year ON board_games_embeddings(year_published);
CREATE INDEX IF NOT EXISTS idx_board_games_rating ON board_games_embeddings(rating_average);
CREATE INDEX IF NOT EXISTS idx_board_games_rank ON board_games_embeddings(bgg_rank);
CREATE INDEX IF NOT EXISTS idx_board_games_players ON board_games_embeddings(min_players, max_players);
CREATE INDEX IF NOT EXISTS idx_board_games_play_time ON board_games_embeddings(play_time);
CREATE INDEX IF NOT EXISTS idx_board_games_age ON board_games_embeddings(min_age);
CREATE INDEX IF NOT EXISTS idx_board_games_complexity ON board_games_embeddings(complexity_average);

-- Create vector similarity search index (HNSW for better performance)
CREATE INDEX IF NOT EXISTS idx_board_games_embedding_hnsw 
ON board_games_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_board_games_embeddings_updated_at 
    BEFORE UPDATE ON board_games_embeddings 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE board_games_embeddings IS 'Table storing board game information with vector embeddings for similarity search';
COMMENT ON COLUMN board_games_embeddings.bgg_id IS 'BoardGameGeek unique identifier';
COMMENT ON COLUMN board_games_embeddings.name IS 'Game name';
COMMENT ON COLUMN board_games_embeddings.year_published IS 'Year the game was published';
COMMENT ON COLUMN board_games_embeddings.min_players IS 'Minimum number of players';
COMMENT ON COLUMN board_games_embeddings.max_players IS 'Maximum number of players';
COMMENT ON COLUMN board_games_embeddings.play_time IS 'Playing time in minutes';
COMMENT ON COLUMN board_games_embeddings.min_age IS 'Minimum recommended age';
COMMENT ON COLUMN board_games_embeddings.users_rated IS 'Number of users who rated the game';
COMMENT ON COLUMN board_games_embeddings.rating_average IS 'Average rating (0-10 scale)';
COMMENT ON COLUMN board_games_embeddings.bgg_rank IS 'BoardGameGeek ranking';
COMMENT ON COLUMN board_games_embeddings.complexity_average IS 'Average complexity rating (0-5 scale)';
COMMENT ON COLUMN board_games_embeddings.owned_users IS 'Number of users who own the game';
COMMENT ON COLUMN board_games_embeddings.mechanics IS 'Game mechanics (comma-separated)';
COMMENT ON COLUMN board_games_embeddings.domains IS 'Game domains/categories (comma-separated)';
COMMENT ON COLUMN board_games_embeddings.url IS 'BoardGameGeek URL';
COMMENT ON COLUMN board_games_embeddings.description IS 'Game description';
COMMENT ON COLUMN board_games_embeddings.embedding IS 'Vector embedding for similarity search (1536 dimensions)';

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON board_games_embeddings TO your_app_user;
-- GRANT USAGE ON SEQUENCE board_games_embeddings_id_seq TO your_app_user;
