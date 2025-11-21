import aiosqlite
import json
import os

DATABASE_FILE = os.path.join(os.path.dirname(__file__), 'hair_color_profiles.db')

async def init_db():
    """Initializes the SQLite database and creates the trained_colors table."""
    print("Initializing database...")
    async with aiosqlite.connect(DATABASE_FILE) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trained_colors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                color_name TEXT NOT NULL,
                lab_color_1 TEXT NOT NULL,
                lab_color_2 TEXT NOT NULL,
                lab_color_3 TEXT NOT NULL,
                source_image_filename TEXT
            )
        """)
        await db.commit()
    print("Database initialized.")

async def add_trained_color(name: str, lab_colors: list, filename: str):
    """Adds a new trained color profile to the database."""
    # Ensure lab_colors are stored as JSON strings
    lab_colors_json = [json.dumps(list(c)) for c in lab_colors]
    # Pad with nulls if fewer than 3 colors
    while len(lab_colors_json) < 3:
        lab_colors_json.append(json.dumps([0,0,0]))

    async with aiosqlite.connect(DATABASE_FILE) as db:
        await db.execute(
            "INSERT INTO trained_colors (color_name, lab_color_1, lab_color_2, lab_color_3, source_image_filename) VALUES (?, ?, ?, ?, ?)",
            (name, lab_colors_json[0], lab_colors_json[1], lab_colors_json[2], filename)
        )
        await db.commit()

async def get_all_trained_colors():
    """Retrieves all trained color profiles from the database."""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        cursor = await db.execute("SELECT id, color_name, lab_color_1, lab_color_2, lab_color_3, source_image_filename FROM trained_colors")
        rows = await cursor.fetchall()
        trained_colors = []
        for row in rows:
            trained_colors.append({
                "id": row[0],
                "name": row[1],
                "lab_colors": [json.loads(row[2]), json.loads(row[3]), json.loads(row[4])],
                "source_image": row[5]
            })
        return trained_colors

async def clear_all_data():
    """Clears all trained color profiles from the database."""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        await db.execute("DELETE FROM trained_colors")
        await db.commit()