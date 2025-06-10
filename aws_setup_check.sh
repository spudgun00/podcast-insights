#!/bin/bash
# Check and setup AWS credentials for ECR access

echo "🔐 Checking AWS setup for ECR access..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not installed"
    echo "Install with: brew install awscli"
    exit 1
fi

# Check AWS credentials
if aws sts get-caller-identity &>/dev/null; then
    echo "✅ AWS credentials configured"
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION=$(aws configure get region || echo $AWS_REGION || echo "eu-west-2")
    echo "   Account: $ACCOUNT_ID"
    echo "   Region: $REGION"
    
    # Check ECR access
    echo ""
    echo "🔍 Testing ECR access..."
    if aws ecr describe-repositories --repository-names podcast-ingestor --region $REGION &>/dev/null; then
        echo "✅ ECR repository 'podcast-ingestor' accessible"
        
        # Get ECR login
        echo ""
        echo "🔑 Getting ECR login token..."
        aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
        
        if [ $? -eq 0 ]; then
            echo "✅ ECR login successful"
            echo ""
            echo "🚀 You're ready to build and push!"
            echo "Run: ./test_before_production.sh"
        else
            echo "❌ ECR login failed"
            exit 1
        fi
    else
        echo "❌ Cannot access ECR repository 'podcast-ingestor'"
        echo "Check your permissions or repository name"
        exit 1
    fi
else
    echo "❌ AWS credentials not configured"
    echo ""
    echo "Choose one option:"
    echo ""
    echo "Option 1 - AWS SSO (recommended):"
    echo "  aws configure sso"
    echo "  aws sso login"
    echo ""
    echo "Option 2 - AWS credentials:"
    echo "  aws configure"
    echo "  # Enter your Access Key ID and Secret Access Key"
    echo ""
    echo "Option 3 - Environment variables:"
    echo "  export AWS_ACCESS_KEY_ID=your_key"
    echo "  export AWS_SECRET_ACCESS_KEY=your_secret"
    echo "  export AWS_REGION=eu-west-2"
    exit 1
fi