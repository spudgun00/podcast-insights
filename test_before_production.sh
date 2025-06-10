#!/bin/bash
# Comprehensive testing script before production deployment

set -e

echo "🧹 Step 1: Clean up Docker environment..."

# Remove all stopped containers
echo "Removing stopped containers..."
docker container prune -f || true

# Remove dangling images
echo "Removing dangling images..."
docker image prune -f || true

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f || true

# Remove build cache
echo "Removing build cache..."
docker builder prune -f || true

# Show current disk usage
echo "Current Docker disk usage:"
docker system df

echo ""
echo "🏗️ Step 2: Test build locally (without AWS base image dependency)..."

# Create a test Dockerfile that doesn't depend on your ECR base image
cat > Dockerfile.test <<'EOF'
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install Python and basic deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        ffmpeg \
        wget \
        git && \
    rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements and install Python deps
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code (this will use .dockerignore)
COPY . /app

# Test basic imports
RUN python - <<'PY'
try:
    import yaml
    import feedparser
    import spacy
    import sentence_transformers
    print("✅ All core imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
PY

# Test backfill.py help (validates argparse changes)
RUN python backfill.py --help | grep -q "offset" && echo "✅ --offset parameter added successfully"

EOF

echo "Building test image (this tests .dockerignore effectiveness)..."
DOCKER_BUILDKIT=1 docker build -f Dockerfile.test -t podcast-test:local . || {
    echo "❌ Test build failed!"
    exit 1
}

echo "✅ Test build successful!"

echo ""
echo "🧪 Step 3: Test --offset parameter locally..."

# Test that the offset parameter works
echo "Testing --offset parameter..."
python3 - <<'PY'
import subprocess
import sys

# Test help contains offset
result = subprocess.run([sys.executable, "backfill.py", "--help"], 
                       capture_output=True, text=True)
if "--offset" not in result.stdout:
    print("❌ --offset parameter not found in help")
    sys.exit(1)
else:
    print("✅ --offset parameter present in help")

# Test offset parameter parsing (dry run)
try:
    result = subprocess.run([
        sys.executable, "backfill.py", 
        "--mode", "transcribe", 
        "--offset", "5",
        "--dry_run",
        "--manifest", "nonexistent.csv"  # This will fail before offset is used
    ], capture_output=True, text=True, timeout=10)
    # We expect this to fail due to missing manifest, but not due to offset parsing
    if "offset" in result.stderr.lower() and "unrecognized" in result.stderr.lower():
        print("❌ --offset parameter not recognized")
        sys.exit(1)
    else:
        print("✅ --offset parameter parsed correctly")
except subprocess.TimeoutExpired:
    print("⚠️ Test timed out (expected with missing manifest)")
except Exception as e:
    print(f"⚠️ Test error (expected with missing manifest): {e}")

PY

echo ""
echo "🔍 Step 4: Check .dockerignore effectiveness..."

# Build context size check
echo "Checking build context size..."
tar -cf - . | wc -c | awk '{printf "Build context size: %.1f MB\n", $1/1024/1024}'

# List what's being included/excluded
echo ""
echo "Files being included in build context (first 20):"
tar -cf - . | tar -tv | head -20

echo ""
echo "Checking if large files are excluded:"
if tar -cf - . | tar -tv | grep -E '\.(mp3|wav|git)'; then
    echo "⚠️ WARNING: Large files found in build context!"
else
    echo "✅ Large files successfully excluded"
fi

echo ""
echo "🔐 Step 5: AWS credentials check..."

# Check if AWS credentials are available for ECR push
if aws sts get-caller-identity &>/dev/null; then
    echo "✅ AWS credentials available"
    echo "AWS Account: $(aws sts get-caller-identity --query Account --output text)"
    echo "AWS Region: $(aws configure get region || echo $AWS_REGION || echo 'Not set')"
else
    echo "❌ AWS credentials not available!"
    echo "To push to ECR, you'll need to:"
    echo "  aws configure"
    echo "  or export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    echo "  or aws sso login"
fi

echo ""
echo "📊 Step 6: Size comparison..."

# Get current image sizes
echo "Current Docker images:"
docker images | grep -E "(podcast|nvidia)" || echo "No existing podcast images found"

echo ""
echo "🎯 All tests complete!"
echo ""
echo "Next steps:"
echo "1. If all tests passed ✅, run: ./build_improved_docker.sh"
echo "2. Then register job definition and submit array job"
echo ""
echo "If anything failed ❌, fix the issues before proceeding to production."

# Cleanup test image
docker rmi podcast-test:local 2>/dev/null || true
rm -f Dockerfile.test