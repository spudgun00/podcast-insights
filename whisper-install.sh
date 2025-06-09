#!/bin/bash
yum update -y
pip3 install --upgrade pip
pip3 install openai-whisper torch torchaudio

# Create whisper cache directory
mkdir -p /home/ec2-user/.cache/whisper
chown ec2-user:ec2-user /home/ec2-user/.cache/whisper

# Install models as ec2-user
sudo -u ec2-user python3 -c "
import whisper
print('Downloading large-v3...')
model_large = whisper.load_model('large-v3')
print('✅ Large-v3 downloaded and cached')
print('Downloading medium...')
model_medium = whisper.load_model('medium')
print('✅ Medium downloaded and cached')
print('Installation complete!')
"

# Verify installation
ls -lh /home/ec2-user/.cache/whisper/
du -sh /home/ec2-user/.cache/whisper/*

# Clean up
yum clean all
echo '✅ Whisper installation complete' > /tmp/whisper-ready
