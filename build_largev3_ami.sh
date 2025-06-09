#!/usr/bin/env bash

set -euo pipefail

# --------- FILL THESE IN --------------
KEY="podcast-ingester"              # key-pair name (no .pem)
SUBNET="subnet-03ff99569a917a252"   # public subnet
SG="sg-0abc123de45f67890"           # SG that allows SSH from your IP
ROLE="pod-insight-batch-role"       # instance-profile name
ACCOUNT="594331569440"              # your AWS Account ID
# --------------------------------------

# 1. Launch a temporary g5.xlarge Spot
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-047d0f3b4f26c884f \
  --instance-type g5.xlarge \
  --instance-market-options MarketType=spot \
  --key-name "$KEY" \
  --iam-instance-profile Name=$ROLE \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":60,"VolumeType":"gp3"}}]' \
  --security-group-ids "$SG" \
  --subnet-id "$SUBNET" \
  --query 'Instances[0].InstanceId' --output text)

echo "Instance $INSTANCE_ID launching …  (this takes ~90 s)"
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"

IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Public IP = $IP"

# 2. Push setup commands to the instance
ssh -o StrictHostKeyChecking=no ubuntu@"$IP" <<EOF
set -e
sudo apt-get update -y && sudo apt-get install -y git-lfs

sudo mkdir -p /opt/whisper && sudo chown \$USER /opt/whisper
cd /opt/whisper
echo "Downloading Whisper Large-v3 weights …"
wget -q https://huggingface.co/openai/whisper-large-v3/resolve/main/v3-large.pt -O large-v3.bin

echo "Pulling pipeline container layers (for cache) …"
aws ecr get-login-password --region eu-west-2 | \
  docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.eu-west-2.amazonaws.com
docker pull ${ACCOUNT}.dkr.ecr.eu-west-2.amazonaws.com/podinsight/transcribe:latest

echo "De-provisioning instance and powering off …"
sudo /usr/sbin/waagent -deprovision+user -force
sudo shutdown -h now
EOF

# 3. Create the AMI from the stopped box
AMI_ID=$(aws ec2 create-image \
           --instance-id "$INSTANCE_ID" \
           --name "pod-insight-dlami-largev3-$(date +%Y%m%d-%H%M)" \
           --no-reboot \
           --query 'ImageId' --output text)
echo "✅ AMI snapshot started: $AMI_ID   (shows as 'pending' for ~2-3 min)"

# 4. Terminate the temporary builder
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
echo "Builder instance terminated."