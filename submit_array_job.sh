#!/bin/bash
# Script to submit the corrected array job for transcribing 185 episodes

set -e

echo "🚀 Submitting AWS Batch array job for 185 episode transcription..."

# Submit the array job
aws batch submit-job \
  --job-name "podinsight-transcribe-185-episodes-$(date +%Y%m%d-%H%M%S)" \
  --job-queue "podinsight-transcribe-queue" \
  --job-definition "podinsight_transcribe_array_20250610" \
  --array-properties "size=185" \
  --region eu-west-2

echo "✅ Array job submitted successfully!"
echo ""
echo "📊 Monitoring commands:"
echo "  - List jobs: aws batch list-jobs --job-queue podinsight-transcribe-queue --region eu-west-2"
echo "  - Monitor costs: python cost_analyzer.py"
echo "  - Emergency stop: python emergency_shutdown.py"
echo ""
echo "🎯 Expected completion: 2-3 hours with up to 22 concurrent GPUs"
echo "💰 Estimated cost: ~$25-30 for all 185 episodes"