#!/bin/bash
# Quick rebuild and test script

set -e

echo "🧹 Step 1: Clean up Docker to free space"
docker system prune -f --volumes
docker builder prune -f
echo "✅ Docker cleanup complete"

echo ""
echo "🔑 Step 2: AWS credentials and ECR login"
ACCOUNT_ID="594331569440"
REGION="eu-west-2"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
echo "✅ ECR login successful"

echo ""
echo "🐳 Step 3: Building updated Docker image with automatic array index detection..."
IMAGE_TAG="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/podcast-ingestor:medium-fp16-cudnn9-v4"

docker build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t $IMAGE_TAG \
    -f Dockerfile.improved \
    .

echo "✅ Docker build completed"

echo ""
echo "🚀 Step 4: Pushing to ECR..."
docker push $IMAGE_TAG
echo "✅ Image pushed to ECR"

echo ""
echo "🔧 Step 5: Updating job definition to use new image..."
sed -i.bak "s|medium-fp16-cudnn9-v3|medium-fp16-cudnn9-v4|g" updated_job_definition.json
echo "✅ Job definition updated"

echo ""
echo "📝 Step 6: Registering job definition..."
aws batch register-job-definition \
  --cli-input-json file://updated_job_definition.json \
  --region $REGION
echo "✅ Job definition registered"

echo ""
echo "🧪 Step 7: Submitting test job..."
TEST_JOB_NAME="test-auto-index-$(date +%Y%m%d-%H%M%S)"

aws batch submit-job \
  --job-name "$TEST_JOB_NAME" \
  --job-queue "podinsight_smoke_q_20250606" \
  --job-definition "podinsight_small_run_20250610" \
  --array-properties "size=6" \
  --region $REGION

echo "✅ Test job submitted: $TEST_JOB_NAME"

echo ""
echo "🎯 What this test will do:"
echo "  - Index 0-4: Skip (already completed episodes)"
echo "  - Index 5: Process new episode (automatic array index detection)"
echo ""
echo "📊 Monitor with:"
echo "  aws batch list-jobs --job-queue podinsight_smoke_q_20250606 --region $REGION"