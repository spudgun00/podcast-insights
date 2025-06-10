#!/bin/bash
# Test with your existing infrastructure

set -e

echo "🔧 Updating Job Definition with improved image and correct command..."

# Register the updated job definition (creates new revision)
aws batch register-job-definition \
  --cli-input-json file://updated_job_definition.json \
  --region eu-west-2

echo "✅ Job definition updated to new revision!"

# Submit single test job using your existing queue
TEST_JOB_NAME="test-improved-$(date +%Y%m%d-%H%M%S)"

echo "🧪 Submitting test job: $TEST_JOB_NAME"
echo "Using:"
echo "  Queue: podinsight_smoke_q_20250606"
echo "  Job Definition: podinsight_small_run_20250610 (latest revision)"
echo "  Array size: 6 (test skip logic + process 1 new episode)"

aws batch submit-job \
  --job-name "$TEST_JOB_NAME" \
  --job-queue "podinsight_smoke_q_20250606" \
  --job-definition "podinsight_small_run_20250610" \
  --array-properties "size=6" \
  --region eu-west-2

echo ""
echo "✅ Test job submitted!"
echo ""
echo "📊 Monitor with:"
echo "  aws batch list-jobs --job-queue podinsight_smoke_q_20250606 --region eu-west-2"
echo ""
echo "🎯 What this test will do:"
echo "  - Index 0-4: Skip (already completed episodes) - tests skip logic ✅"
echo "  - Index 5: Process new episode (d8d0ea45-0536-41b9-aeff-cb2998c488f8) 🚀"
echo ""
echo "🎯 Expected improvements:"
echo "  - ⚡ Faster startup (no 360MB model downloads)"
echo "  - 🧹 Cleaner logs (fewer deprecation warnings)"
echo "  - 💰 Same cost, better performance"
echo ""
echo "If test succeeds, run full 185-episode batch with:"
echo "  aws batch submit-job --job-name 'full-185-episodes' --job-queue 'podinsight_smoke_q_20250606' --job-definition 'podinsight_small_run_20250610' --array-properties 'size=185' --region eu-west-2"