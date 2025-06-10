#!/bin/bash
# Complete build and test script

set -e

echo "🔍 Step 1: Test local changes with NO_AWS=1"
echo "Testing backfill.py offset parameter..."

# Quick local test of offset functionality
NO_AWS=1 python backfill.py --mode transcribe --manifest /dev/null --limit 1 --offset 5 --dry_run || {
    echo "❌ backfill.py offset test failed"
    exit 1
}

echo "✅ Local offset parameter works"

echo ""
echo "🔑 Step 2: Setup AWS credentials and ECR login"

# Check AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "❌ AWS credentials not configured"
    echo "Run: aws configure sso && aws sso login"
    exit 1
fi

# Get ECR login
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region || echo "eu-west-2")
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

echo "✅ AWS credentials and ECR access confirmed"

echo ""
echo "🐳 Step 3: Build improved Docker image"

# Build with the improved Dockerfile
IMAGE_TAG="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/podcast-ingestor:medium-fp16-cudnn9-v3"

echo "Building: $IMAGE_TAG"
docker build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t $IMAGE_TAG \
    -f Dockerfile.improved \
    .

echo "✅ Docker build completed"

echo ""
echo "🚀 Step 4: Push to ECR"
docker push $IMAGE_TAG

echo "✅ Image pushed to ECR"

echo ""
echo "🧪 Step 5: Update job definition with new image"

# Update the job definition to use the new image
sed -i.bak "s|medium-fp16-cudnn9-v2|medium-fp16-cudnn9-v3|g" updated_job_definition.json

echo "✅ Job definition updated"

echo ""
echo "📋 Next steps:"
echo "1. Register updated job definition:"
echo "   aws batch register-job-definition --cli-input-json file://updated_job_definition.json --region $REGION"
echo ""
echo "2. Submit test job:"
echo "   aws batch submit-job --job-name 'test-v3-\$(date +%Y%m%d-%H%M%S)' --job-queue 'podinsight_smoke_q_20250606' --job-definition 'podinsight_small_run_20250610' --array-properties 'size=6' --region $REGION"
echo ""
echo "3. Monitor:"
echo "   aws batch list-jobs --job-queue podinsight_smoke_q_20250606 --region $REGION"