#!/bin/bash
# Complete Docker build, push, and test script

set -e

echo "🧹 Step 1: Clean up Docker to free space"
echo "Current Docker usage:"
docker system df

echo ""
echo "Cleaning up..."
docker system prune -f --volumes
docker builder prune -f
echo "✅ Docker cleanup complete"

echo ""
echo "🔑 Step 2: AWS credentials and ECR login"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="eu-west-2"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
echo "✅ ECR login successful"

echo ""
echo "🐳 Step 3: Build improved Docker image"
IMAGE_TAG="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/podcast-ingestor:medium-fp16-cudnn9-v3"

echo "Building: $IMAGE_TAG"
docker build \
    --no-cache \
    -t $IMAGE_TAG \
    -f Dockerfile.improved \
    .

echo "✅ Docker build completed"

echo ""
echo "🚀 Step 4: Push to ECR"
docker push $IMAGE_TAG
echo "✅ Image pushed to ECR"

echo ""
echo "🔧 Step 5: Update job definition"
# Fix job definition to use correct podrun.py command structure
cat > updated_job_definition.json << 'EOF'
{
  "jobDefinitionName": "podinsight_small_run_20250610",
  "type": "container",
  "timeout": {
    "attemptDurationSeconds": 9000
  },
  "containerProperties": {
    "image": "594331569440.dkr.ecr.eu-west-2.amazonaws.com/podcast-ingestor:medium-fp16-cudnn9-v3",
    "vcpus": 4,
    "memory": 15600,
    "command": [
      "transcribe",
      "--manifest",
      "s3://pod-insights-manifests/20250526_manifest.csv",
      "--model",
      "medium",
      "--limit",
      "1",
      "--offset",
      "${AWS_BATCH_JOB_ARRAY_INDEX}"
    ],
    "volumes": [],
    "environment": [
      {
        "name": "AWS_REGION",
        "value": "eu-west-2"
      },
      {
        "name": "AWS_DEFAULT_REGION",
        "value": "eu-west-2"
      },
      {
        "name": "WHISPER_COMPUTE_TYPE",
        "value": "float16"
      }
    ],
    "mountPoints": [],
    "ulimits": [],
    "resourceRequirements": [
      {
        "value": "1",
        "type": "GPU"
      }
    ],
    "secrets": []
  },
  "platformCapabilities": [
    "EC2"
  ]
}
EOF

echo "✅ Job definition updated with correct podrun.py command"

echo ""
echo "📝 Step 6: Register job definition"
aws batch register-job-definition \
  --cli-input-json file://updated_job_definition.json \
  --region $REGION

echo "✅ Job definition registered"

echo ""
echo "🧪 Step 7: Submit test job"
TEST_JOB_NAME="test-fixed-podrun-$(date +%Y%m%d-%H%M%S)"

aws batch submit-job \
  --job-name "$TEST_JOB_NAME" \
  --job-queue "podinsight_smoke_q_20250606" \
  --job-definition "podinsight_small_run_20250610" \
  --array-properties "size=6" \
  --region $REGION

echo "✅ Test job submitted: $TEST_JOB_NAME"

echo ""
echo "📊 Monitor with:"
echo "  aws batch list-jobs --job-queue podinsight_smoke_q_20250606 --region $REGION"

echo ""
echo "🎯 Test expectations:"
echo "  - Index 0-4: Skip (already completed) - tests skip logic"
echo "  - Index 5: Process new episode - tests --offset parameter"
echo ""
echo "If successful, run full 185-episode batch with:"
echo "  aws batch submit-job --job-name 'full-185-episodes' --job-queue 'podinsight_smoke_q_20250606' --job-definition 'podinsight_small_run_20250610' --array-properties 'size=185' --region $REGION"