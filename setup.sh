#!/bin/bash
set -e

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up podcast-insights environment...${NC}"

# Clean up any existing environments (optional)
echo -e "${YELLOW}Checking for existing environments...${NC}"
if [ -d "fresh-env" ]; then
    echo -e "${YELLOW}Removing previous fresh-env...${NC}"
    rm -rf fresh-env
fi

# Create virtual environment
echo -e "${GREEN}Creating virtual environment...${NC}"
python3 -m venv .venv --prompt podcast-env

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo -e "${GREEN}Creating data directories...${NC}"
mkdir -p data/audio
mkdir -p data/transcripts
mkdir -p data/insights

echo -e "${GREEN}Setup complete! Activate the environment with:${NC}"
echo -e "${YELLOW}source .venv/bin/activate${NC}"
echo -e "${GREEN}Run a test with:${NC}"
echo -e "${YELLOW}NO_AWS=1 python backfill_test.py --feed https://thetwentyminutevc.libsyn.com/rss --limit 1 --model_size base${NC}" 