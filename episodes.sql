-- episodes.sql  (run once in psql)
CREATE TABLE IF NOT EXISTS episodes (
    guid              TEXT PRIMARY KEY,
    podcast_slug      TEXT NOT NULL,
    podcast_title     TEXT NOT NULL,
    episode_title     TEXT NOT NULL,
    published_date    DATE NOT NULL,
    slug              TEXT UNIQUE,               -- "20vc-2025-05-22-chime-ipo"
    s3_prefix         TEXT NOT NULL,

    -- handy pointers; NULL until file exists
    meta_s3_path              TEXT,
    transcript_s3_path        TEXT,
    cleaned_entities_s3_path  TEXT,

    duration_sec      INTEGER,
    asr_engine        TEXT,                      -- e.g. 'whisperx|base|ct2'
    processed_at      TIMESTAMPTZ DEFAULT NOW(),
    last_updated_at   TIMESTAMPTZ DEFAULT NOW()
);

-- speed up look-ups by podcast + date
CREATE INDEX IF NOT EXISTS idx_show_day
    ON episodes (podcast_slug, published_date); 