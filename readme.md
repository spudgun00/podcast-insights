# Podcast Insights

A system for downloading, transcribing, and extracting insights from podcasts.

## Setup

1. Clone this repository
2. Run the setup script:
   ```
   ./setup.sh
   ```
   
This will create a virtual environment, install all required dependencies, and set up necessary directories.

## Usage

### Activate the environment
```
source .venv/bin/activate
```

### Download and transcribe podcasts
```
export NO_AWS=1 && python3 backfill_test.py --since 2023-05-01 --limit 2 --model_size base
```

Parameters:
- `--since`: Date to start downloading podcasts from (YYYY-MM-DD)
- `--limit`: Maximum number of episodes to process
- `--model_size`: Whisper model size (tiny, base, small, medium, large)

### Check timestamp support for podcast URLs
```
python3 check_timestamp_support.py
```

This will check if the podcast URLs support timestamp fragments, which is useful for linking to specific parts of episodes.

### Update podcast metadata
```
python3 patch_meta_enhanced.py
```

This will update the metadata for podcasts, including episode information, timestamps, etc.

## AWS Deployment

To deploy on AWS:
1. Remove the `NO_AWS=1` environment variable
2. Configure AWS credentials
3. Update the configuration in `config.yaml` as needed

## Directory Structure

- `/tmp/audio`: Downloaded podcast audio files
- `/tmp/transcripts`: Generated transcripts

## Requirements

All requirements are listed in `requirements.txt` and will be installed by the setup script.
