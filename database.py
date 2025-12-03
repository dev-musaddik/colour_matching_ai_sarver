import aiosqlite as aqlite
import os

DB_FILE = "hair_color_analyzer.db"

async def initialize_database():
    """
    Initializes the main application database and creates the necessary tables
    for the training-based color learning platform.
    """
    async with aqlite.connect(DB_FILE) as db:
        # Table for user-defined colors
        await db.execute("""
            CREATE TABLE IF NOT EXISTS colors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table to store paths to training images associated with a color
        await db.execute("""
            CREATE TABLE IF NOT EXISTS training_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                color_id INTEGER NOT NULL,
                image_path TEXT NOT NULL UNIQUE,
                is_processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (color_id) REFERENCES colors (id) ON DELETE CASCADE
            )
        """)

        # Table to store extracted features for each image
        await db.execute("""
            CREATE TABLE IF NOT EXISTS image_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                lab_features TEXT NOT NULL, -- JSON string of LAB color clusters
                embedding BLOB NOT NULL, -- Storing feature embedding as a blob
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES training_images (id) ON DELETE CASCADE
            )
        """)

        # Table to store trained models for each color
        await db.execute("""
            CREATE TABLE IF NOT EXISTS color_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                color_id INTEGER NOT NULL,
                model_type TEXT NOT NULL, -- e.g., 'svm', 'knn', 'neural_network'
                model_path TEXT NOT NULL UNIQUE,
                version INTEGER NOT NULL DEFAULT 1,
                performance_metrics TEXT, -- JSON string for accuracy, precision, etc.
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (color_id) REFERENCES colors (id) ON DELETE CASCADE
            )
        """)

        await db.commit()

def get_db_path():
    """Returns the absolute path to the database file."""
    return os.path.abspath(DB_FILE)

