# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Environment Setup
```bash
./setup.sh                           # Create venv, install deps, create directories
source .venv/bin/activate            # Activate environment
```

### Local Development (NO_AWS=1)
```bash
# Test single episode processing
NO_AWS=1 python backfill_test.py --since 2023-05-01 --limit 2 --model_size base

# Test specific feed
NO_AWS=1 python backfill_test.py --feed https://thetwentyminutevc.libsyn.com/rss --limit 1 --model_size base

# Test transcription only
NO_AWS=1 python transcribe.py path/to/audio.mp3
```

### Production AWS Mode
```bash
# Two-phase pipeline via CLI
podrun fetch --limit 50 --dry-run     # Generate manifest + download (test)
podrun fetch --limit 1000             # Download phase (CPU)
podrun transcribe --limit 1000 --manifest manifest.csv --model base  # Transcription phase (GPU)

# Direct backfill.py usage
python backfill.py --mode fetch --manifest manifest.csv --limit 1000
python backfill.py --mode transcribe --manifest manifest.csv --limit 1000 --model_size base
```

### Testing
```bash
pytest tests/                        # Run all tests
pytest tests/test_meta_utils.py -v   # Test people normalization
pytest tests/test_kpi_utils.py -v    # Test KPI extraction
```

### Utilities
```bash
python tools/generate_manifest.py --config tier1_feeds.yaml --output-csv manifest.csv
python patch_meta_enhanced.py        # Update existing metadata
```

## Core Architecture

### Two-Phase Processing Pipeline

**Phase 1 - CPU Download** (`--mode fetch`):
- Processes RSS feeds → manifest.csv → download MP3s + minimal metadata  
- Stores in S3 `pod-insights-raw` or local `data/audio`
- Uses host-specific throttling to respect CDN rate limits
- AWS: 80 parallel Fargate Spot tasks (~$4.85 for 1000 episodes)

**Phase 2 - GPU Transcription** (`--mode transcribe`):
- Reads audio from Phase 1 → WhisperX transcription → metadata enrichment
- Stores in S3 `pod-insights-stage` or local `data/transcripts`  
- AWS: 10 GPU instances (g5.xlarge Spot, ~$24.9 for 1000 episodes)

### Dual Operation Modes

**AWS Mode** (production): DynamoDB + S3 + CloudWatch metrics  
**Local Mode** (`NO_AWS=1`): PostgreSQL + local files + no AWS calls

The system automatically detects mode via the `NO_AWS` environment variable and uses appropriate storage backends.

### S3 Storage Layout
```
pod-insights-raw/<feed_slug>/<guid>/audio/episode.mp3
                                   /meta/meta.json

pod-insights-stage/<feed_slug>/<guid>/transcripts/transcript.json
                                     /segments/segments.json  
                                     /kpis/kpis.json
                                     /embeddings/*.npy
```

### Key Modules

**Storage & Infrastructure:**
- `settings.py` - S3 bucket configuration, path layout functions
- `s3_utils.py` - S3 operations, URI parsing
- `db_utils.py` (PostgreSQL) / `db_utils_dynamo.py` (DynamoDB) - Episode status tracking
- `metrics.py` - CloudWatch metrics publishing

**Processing Pipeline:**
- `feed_utils.py` - RSS parsing with polite user-agent
- `audio_utils.py` - MP3 download with retry and host throttling
- `transcribe.py` - WhisperX integration
- `meta_utils.py` - Entity extraction, people normalization via `config/people_aliases.yml`
- `kpi_utils.py` - Insight extraction from transcripts

### Configuration Files

- `tier1_feeds.yaml` - Production feed list with date filtering
- `config.yaml` - Basic local development settings  
- `config/people_aliases.yml` - Guest/host name normalization
- `manifest.csv` - Generated episode list driving both processing phases

### Error Handling Patterns

The codebase uses a `_NoAWS` shim pattern to gracefully handle missing AWS services in local mode. Most modules check the `NO_AWS` environment variable and switch between AWS and local implementations.

### Host Throttling

Uses `threading.Semaphore` with per-hostname limits to prevent rate limiting from podcast CDNs:
```python
HOST_LIMIT = {"megaphone.fm": 5, "libsyn.com": 3, "simplecast.com": 4}
```

### Metadata Enrichment Pipeline

1. **Download**: Basic RSS metadata extraction
2. **Transcription**: WhisperX speech-to-text with speaker diarization  
3. **Entity Extraction**: spaCy NLP + custom people normalization
4. **KPI Generation**: Highlights, insights, sentiment analysis
5. **Embeddings**: sentence-transformers for semantic search