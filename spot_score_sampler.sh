#!/usr/bin/env bash
# File: spot_score_sampler.sh
# Collect Spot placement scores for g5.xlarge every time you run it.
# Tested with AWS CLI v2.12+

set -euo pipefail

# ── CONFIG ──────────────────────────────────────────
REGIONS=(eu-west-2 eu-west-1 us-east-1 eu-east-2 eu-north-2)
         # London, Ireland, New York, N. Virginia,
TARGETS=(1 5 10 20)                      # 20 removed – API hard-limit
INST_TYPE="g5.xlarge"
LOG="$HOME/spot_placement_scores.log"
# ────────────────────────────────────────────────────

# Header once
[[ -f $LOG ]] || printf "timestamp,region,target_cap,score\n" >"$LOG"

for R in "${REGIONS[@]}"; do
  for T in "${TARGETS[@]}"; do
    RAW=$(aws ec2 get-spot-placement-scores \
            --instance-types "$INST_TYPE"  \
            --target-capacity "$T"         \
            --region "$R" --output json 2>/dev/null)

    SCORE=$(jq -r '.SpotPlacementScores[0].Score // "NA"' <<<"$RAW")

    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")        # portable ISO-8601
    printf "%s,%s,%s,%s\n" "$TS" "$R" "$T" "$SCORE" >>"$LOG"
    echo "[$R] cap=$T → score=$SCORE"
  done
done