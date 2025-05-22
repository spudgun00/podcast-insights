import os
import psycopg2
import psycopg2.extras
import datetime
import logging

logger = logging.getLogger(__name__)

# Default to a local setup. For AWS, this would be set via environment variables.
_URL = os.getenv("PG_URL", "postgresql://insights:insights@localhost:5432/insights")
_DISABLE_DB = os.getenv("DISABLE_DB_OPERATIONS", "false").lower() == "true"

def _conn():
    if _DISABLE_DB:
        logger.info("Database operations are disabled via DISABLE_DB_OPERATIONS.")
        return None
    try:
        return psycopg2.connect(_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    except psycopg2.OperationalError as e:
        logger.error(f"Failed to connect to PostgreSQL database at {_URL}. Error: {e}")
        logger.error("Database operations will be skipped. Ensure PostgreSQL is running and configured correctly.")
        logger.error("You can set DISABLE_DB_OPERATIONS=true to suppress this error if DB is not required.")
        return None

def init_db(sql_file_path="episodes.sql"):
    """Initializes the database by executing DDL from the specified SQL file."""
    conn = _conn()
    if not conn:
        return
    try:
        with open(sql_file_path, 'r') as f:
            ddl = f.read()
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
        logger.info(f"Database initialized successfully using {sql_file_path}.")
    except FileNotFoundError:
        logger.error(f"SQL DDL file not found at {sql_file_path}. Database not initialized.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def upsert_episode(row: dict):
    """
    Upserts an episode record into the database.
    Required keys in row: guid, podcast_slug, podcast_title, episode_title, published_date, s3_prefix.
    Optional keys will be included if present in the row dict.
    """
    conn = _conn()
    if not conn:
        return

    required_keys = ["guid", "podcast_slug", "podcast_title", "episode_title", "published_date", "s3_prefix"]
    for k in required_keys:
        if k not in row or row[k] is None:
            logger.error(f"Missing required key '{k}' for upsert_episode. Row: {row}")
            return

    # Filter out keys from row that are not columns in the episodes table based on a predefined list or introspection
    # For now, let's assume all keys in `row` intended for `episodes` table are valid
    # Or better, define the full list of expected columns explicitly
    allowed_cols = [ # Should match the CREATE TABLE statement + any dynamic ones like 'now'
        "guid", "podcast_slug", "podcast_title", "episode_title", "published_date", "slug", "s3_prefix",
        "meta_s3_path", "transcript_s3_path", "cleaned_entities_s3_path",
        "duration_sec", "asr_engine"
    ]
    
    # Prepare columns and values for SQL statement
    cols_to_insert = []
    vals_to_insert = []
    
    for col_name in allowed_cols:
        if col_name in row:
            cols_to_insert.append(col_name)
            vals_to_insert.append(row[col_name])

    if not cols_to_insert: # Should not happen if required keys are present
        logger.error("No valid columns to insert for upsert_episode.")
        return

    # Construct SET part for ON CONFLICT DO UPDATE
    # Exclude primary key 'guid' from the SET clause
    set_statements = []
    for col_name in cols_to_insert:
        if col_name != "guid":
            set_statements.append(f"{col_name}=EXCLUDED.{col_name}")
    
    # Always update last_updated_at
    set_statements.append("last_updated_at = %(now)s") 
    sets_sql = ", ".join(set_statements)

    sql = f"""
        INSERT INTO episodes ({", ".join(cols_to_insert)})
        VALUES ({", ".join(['%s'] * len(vals_to_insert))})
        ON CONFLICT (guid) DO UPDATE SET {sets_sql};
    """

    # Add 'now' to the values list for the last_updated_at = %(now)s part if using named placeholders
    # For %s placeholders, psycopg2 expects a sequence of values.
    # The original snippet had row["now"] = ... and then vals + [row["now"]]
    # but %s placeholders don't use named dicts directly. We need to ensure 'now' is passed as the last value.
    
    current_time_utc = datetime.datetime.now(datetime.timezone.utc)
    final_values = vals_to_insert + [current_time_utc] 

    # The SQL query now references `%(now)s` in the SET clause, but uses `%s` elsewhere.
    # This is a mix that won't work directly. 
    # Let's adjust the SQL to use %s for the `now` value as well.
    # The SETS part now correctly includes `last_updated_at = %s` if we add it before joining set_statements
    # Or, ensure `now` is part of the row dict that psycopg2 can map if we used RealDictCursor style execution with named placeholders

    # Rebuild SETS part to use %s for `now` directly
    # Create the set_statements list
    set_statements_for_sql = []
    for col_name in cols_to_insert:
        if col_name != "guid": # Exclude the primary key from update
            set_statements_for_sql.append(f"{col_name} = EXCLUDED.{col_name}")
    set_statements_for_sql.append("last_updated_at = %s") # Placeholder for the current timestamp
    
    final_sets_sql = ", ".join(set_statements_for_sql)

    # Ensure vals_to_insert for the INSERT part does not include the timestamp
    # The update part will use values from EXCLUDED and the separately provided timestamp.

    sql_final = f"""
        INSERT INTO episodes ({", ".join(cols_to_insert)})
        VALUES ({", ".join(['%s'] * len(vals_to_insert))})
        ON CONFLICT (guid) DO UPDATE SET {final_sets_sql};
    """
    
    # For the UPDATE part, the values are implicitly from EXCLUDED for most columns.
    # The only explicit value needed for the UPDATE part is the timestamp for last_updated_at.
    # So, the parameters passed to execute should be vals_to_insert + [current_time_utc]
    # This implies the number of %s in final_sets_sql should match the extra values we provide for the UPDATE case.
    # If `final_sets_sql` becomes `col1=EXCLUDED.col1, col2=EXCLUDED.col2, last_updated_at=%s`,
    # then execute(sql_final, vals_to_insert + [current_time_utc]) is correct.

    try:
        with conn.cursor() as cur:
            cur.execute(sql_final, vals_to_insert + [current_time_utc]) # Pass all values for INSERT, and `now` for UPDATE
        conn.commit()
        logger.info(f"Successfully upserted episode GUID: {row['guid']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error upserting episode GUID {row.get('guid', 'UNKNOWN')}: {e}")
        logger.error(f"SQL attempted: {sql_final}")
        logger.error(f"Values passed: {vals_to_insert + [current_time_utc]}")
    finally:
        if conn:
            conn.close()

def guid_for(podcast_slug: str, published_date: str) -> str | None:
    """Retrieves a GUID for a given podcast_slug and published_date (YYYY-MM-DD string)."""
    conn = _conn()
    if not conn:
        return None
    guid = None
    try:
        # Ensure published_date is a valid date string if not already a date object
        # The table expects DATE type. psycopg2 can handle string conversion if in YYYY-MM-DD.
        datetime.date.fromisoformat(published_date) # Validate format

        sql = """SELECT guid FROM episodes
                 WHERE podcast_slug=%s AND published_date=%s;"""
        with conn.cursor() as cur:
            cur.execute(sql, (podcast_slug, published_date))
            res = cur.fetchone()
        if res:
            guid = res["guid"]
            logger.info(f"Found GUID {guid} for {podcast_slug} on {published_date}")
        else:
            logger.info(f"No GUID found for {podcast_slug} on {published_date}")
    except ValueError:
        logger.error(f"Invalid published_date format: '{published_date}'. Expected YYYY-MM-DD.")
    except Exception as e:
        logger.error(f"Error fetching GUID for {podcast_slug} on {published_date}: {e}")
    finally:
        if conn:
            conn.close()
    return guid

# Example usage (typically called from a main script or test)
if __name__ == '__main__':
    # This is for demonstration or direct execution, not typically part of a library module.
    # 1. Initialize DB (run once, or ensure it runs before other operations)
    # Configure logging to see output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check if DB operations are disabled before trying to init
    if not _DISABLE_DB:
        init_db() # Assumes episodes.sql is in the same directory or correct path is given

        # 2. Example upsert
        sample_episode_data = {
            "guid": "test-guid-12345",
            "podcast_slug": "my-fav-podcast",
            "podcast_title": "My Favorite Podcast",
            "episode_title": "Episode 1: The Beginning",
            "published_date": "2023-01-15",
            "slug": "my-fav-podcast-2023-01-15-episode-1-the-beginning",
            "s3_prefix": "s3://bucket/path/to/test-guid-12345/",
            "meta_s3_path": "s3://bucket/path/to/test-guid-12345/meta.json",
            "transcript_s3_path": "s3://bucket/path/to/test-guid-12345/transcript.json",
            "cleaned_entities_s3_path": "s3://bucket/path/to/test-guid-12345/entities_clean.json",
            "duration_sec": 1850,
            "asr_engine": "whisperx|base|ct2"
        }
        upsert_episode(sample_episode_data)

        # 3. Example fetch GUID
        retrieved_guid = guid_for("my-fav-podcast", "2023-01-15")
        if retrieved_guid:
            assert retrieved_guid == sample_episode_data["guid"]
            logger.info(f"Successfully retrieved GUID: {retrieved_guid}")
        
        # Example update (upsert again with same GUID)
        updated_episode_data = sample_episode_data.copy()
        updated_episode_data["episode_title"] = "Episode 1: The Real Beginning (Updated)"
        updated_episode_data["duration_sec"] = 1900
        upsert_episode(updated_episode_data)

        retrieved_guid_updated = guid_for("my-fav-podcast", "2023-01-15")
        if retrieved_guid_updated:
             logger.info(f"GUID after update attempt: {retrieved_guid_updated}")
             # Add assertions here to check if the title/duration updated in DB if you fetch the whole row.
    else:
        logger.info("DB operations disabled, skipping example usage in db_utils.py __main__.") 