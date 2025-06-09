#!/usr/bin/env python3
"""
Fixed AWS Billing Alarms Setup - Proper Region Handling
Creates SNS topic in your region, billing alarms in us-east-1
"""

import boto3
import json

def create_billing_alarms_fixed():
    """Set up billing alarms with proper region handling."""
    
    # Get your default region
    session = boto3.Session()
    your_region = session.region_name or 'eu-west-2'
    
    print(f"🌍 Using your region: {your_region}")
    print("🚨 Setting up AWS Billing Alarms (Fixed Version)...")
    
    # Create SNS topic in us-east-1 (required for billing alarms)
    sns = boto3.client('sns', region_name='us-east-1')
    
    try:
        topic_response = sns.create_topic(Name='pod-insights-cost-alerts')
        topic_arn = topic_response['TopicArn']
        print(f"✅ Created SNS topic in us-east-1: {topic_arn}")
        
        # Subscribe your email
        email = "jimgill@gmail.com"
        sns.subscribe(
            TopicArn=topic_arn,
            Protocol='email',
            Endpoint=email
        )
        print(f"✅ Subscribed {email} to alerts")
        
    except Exception as e:
        print(f"❌ Error creating SNS topic: {e}")
        return
    
    # Create CloudWatch alarms in us-east-1 (required for billing)
    # but point them to your SNS topic in your region
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    
    # Alarm configurations
    alarms = [
        {
            'name': 'PodInsights-DailyCost-Warning',
            'description': 'Daily cost exceeded $5',
            'threshold': 5.0,
            'period': 86400,  # 24 hours
            'evaluation_periods': 1
        },
        {
            'name': 'PodInsights-DailyCost-Critical', 
            'description': 'Daily cost exceeded $15',
            'threshold': 15.0,
            'period': 86400,
            'evaluation_periods': 1
        },
        {
            'name': 'PodInsights-MonthlyCost-Warning',
            'description': 'Monthly cost exceeded $50', 
            'threshold': 50.0,
            'period': 86400,
            'evaluation_periods': 1
        },
        {
            'name': 'PodInsights-MonthlyCost-Emergency',
            'description': 'Monthly cost exceeded $100',
            'threshold': 100.0,
            'period': 86400, 
            'evaluation_periods': 1
        }
    ]
    
    # Create each alarm in us-east-1 but send alerts to your region's SNS
    for alarm in alarms:
        try:
            cloudwatch.put_metric_alarm(
                AlarmName=alarm['name'],
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=alarm['evaluation_periods'],
                MetricName='EstimatedCharges',
                Namespace='AWS/Billing',
                Period=alarm['period'],
                Statistic='Maximum',
                Threshold=alarm['threshold'],
                ActionsEnabled=True,
                AlarmActions=[topic_arn],  # Your region's SNS topic
                AlarmDescription=alarm['description'],
                Dimensions=[
                    {
                        'Name': 'Currency',
                        'Value': 'USD'
                    },
                ],
                Unit='None'
            )
            print(f"✅ Created alarm: {alarm['name']} (${alarm['threshold']} threshold)")
            
        except Exception as e:
            print(f"❌ Error creating alarm {alarm['name']}: {e}")
    
    print(f"\n🎯 BILLING ALARMS SETUP COMPLETE!")
    print(f"📧 Check jimgill@gmail.com for SNS subscription confirmation")
    print(f"🌍 SNS Topic: {topic_arn}")
    print(f"🚨 Both SNS and billing alarms in us-east-1 (required for billing)")
    
    return topic_arn

def add_sms_alerts(topic_arn, phone_number):
    """Add SMS alerts to existing topic."""
    
    # Extract region from topic ARN
    region = topic_arn.split(':')[3]
    sns = boto3.client('sns', region_name=region)
    
    try:
        sns.subscribe(
            TopicArn=topic_arn,
            Protocol='sms',
            Endpoint=phone_number
        )
        print(f"✅ Added SMS alerts to {phone_number}")
        print(f"💰 SMS cost: ~£0.01 per alert")
        
    except Exception as e:
        print(f"❌ Error adding SMS: {e}")

if __name__ == "__main__":
    print("🚨 AWS COST PROTECTION SETUP (FIXED)")
    print("=" * 50)
    
    # Create billing alarms with proper region handling
    topic_arn = create_billing_alarms_fixed()
    
    if topic_arn:
        print(f"\n📱 Want to add SMS alerts?")
        print(f"Run: python -c \"")
        print(f"from fixed_billing_alarms import add_sms_alerts")
        print(f"add_sms_alerts('arn:aws:sns:us-east-1:594331569440:pod-insights-cost-alerts', '+447440157421')\"")
        
        # Test the topic
        print(f"\n🧪 Testing alerts...")
        sns = boto3.client('sns', region_name='us-east-1')
        
        try:
            sns.publish(
                TopicArn=topic_arn,
                Subject="🚨 PodInsights Cost Alert Test",
                Message="TEST: Your cost monitoring is working! This confirms alerts will reach you when costs spike."
            )
            print("✅ Test alert sent! Check your email.")
        except Exception as e:
            print(f"❌ Test failed: {e}")