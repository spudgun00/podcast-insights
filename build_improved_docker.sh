#!/bin/bash
# Script to build and push the improved Docker image

set -e

echo "🚀 Building improved Docker image with optimizations..."

# Enable BuildKit for faster builds and layer caching
export DOCKER_BUILDKIT=1

# Build the improved image
docker build \
  --platform linux/amd64 \
  -f Dockerfile.improved \
  -t 594331569440.dkr.ecr.eu-west-2.amazonaws.com/podcast-ingestor:medium-fp16-cudnn9-v2 \
  .

echo "✅ Docker build complete!"
echo ""
echo "🔄 Pushing to ECR..."

# Push to ECR
docker push 594331569440.dkr.ecr.eu-west-2.amazonaws.com/podcast-ingestor:medium-fp16-cudnn9-v2

echo "✅ Docker push complete!"
echo ""
echo "📊 Improvements made:"
echo "  - Build context reduced by .dockerignore (GB → MB)"
echo "  - Pre-cached SentenceTransformer (eliminates 360MB runtime download)"
echo "  - Pre-cached SpaCy model (eliminates runtime download)"
echo "  - Pre-cached wav2vec2 alignment model (eliminates runtime download)"
echo "  - Added runtime optimizations (OMP_NUM_THREADS, etc.)"
echo "  - Enhanced build-time verification"
echo ""
echo "⚡ Expected improvements:"
echo "  - Push time: ~60min → 5-10min"
echo "  - Job startup: ~4GB downloads → ~0GB downloads"
echo "  - Cleaner logs with fewer warnings"
echo ""
echo "🎯 Ready to submit array job with:"
echo "  ./submit_array_job.sh"