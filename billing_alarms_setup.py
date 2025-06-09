#!/usr/bin/env python3
"""
AWS Billing Alarms Setup
Creates multiple cost monitoring alarms to prevent future cost disasters.
"""

import boto3
import json

def create_billing_alarms():
    """Set up comprehensive billing alarms."""
    
    # Create CloudWatch client in us-east-1 (required for billing metrics)
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    sns = boto3.client('sns', region_name='us-east-1')
    
    print("🚨 Setting up AWS Billing Alarms...")
    
    # Create SNS topic for alerts
    try:
        topic_response = sns.create_topic(Name='pod-insights-cost-alerts')
        topic_arn = topic_response['TopicArn']
        print(f"✅ Created SNS topic: {topic_arn}")
        
        # Subscribe your email (replace with your email)
        email = "founder@podinsighthq.com"  # UPDATE THIS!
        sns.subscribe(
            TopicArn=topic_arn,
            Protocol='email',
            Endpoint=email
        )
        print(f"✅ Subscribed {email} to alerts")
        
    except Exception as e:
        print(f"❌ Error creating SNS topic: {e}")
        return
    
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
    
    # Create each alarm
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
                AlarmActions=[topic_arn],
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
    
    print("\n🎯 BILLING ALARMS SETUP COMPLETE!")
    print("📧 Check your email and confirm the SNS subscription")
    print("🚨 You'll now get alerts if costs spike unexpectedly")

def create_budget():
    """Create AWS Budget for monthly cost tracking."""
    
    budgets = boto3.client('budgets')
    
    budget_config = {
        'BudgetName': 'PodInsights-Monthly-Budget',
        'BudgetLimit': {
            'Amount': '100',  # $100 monthly limit
            'Unit': 'USD'
        },
        'TimeUnit': 'MONTHLY',
        'BudgetType': 'COST',
        'CostFilters': {},
        'TimePeriod': {
            'Start': '2025-06-01',
            'End': '2025-12-31'
        }
    }
    
    notifications = [
        {
            'Notification': {
                'NotificationType': 'ACTUAL',
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': 50,  # Alert at 50% of budget
                'ThresholdType': 'PERCENTAGE'
            },
            'Subscribers': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'founder@podinsighthq.com'  # UPDATE THIS!
                }
            ]
        },
        {
            'Notification': {
                'NotificationType': 'ACTUAL', 
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': 80,  # Alert at 80% of budget
                'ThresholdType': 'PERCENTAGE'
            },
            'Subscribers': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'founder@podinsighthq.com'  # UPDATE THIS!
                }
            ]
        },
        {
            'Notification': {
                'NotificationType': 'FORECASTED',
                'ComparisonOperator': 'GREATER_THAN', 
                'Threshold': 100,  # Alert if forecasted to exceed budget
                'ThresholdType': 'PERCENTAGE'
            },
            'Subscribers': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'founder@podinsighthq.com'  # UPDATE THIS!
                }
            ]
        }
    ]
    
    try:
        budgets.create_budget(
            AccountId='594331569440',  # Your account ID
            Budget=budget_config,
            NotificationsWithSubscribers=notifications
        )
        print("✅ Created monthly budget with alerts")
        
    except Exception as e:
        print(f"❌ Error creating budget: {e}")

if __name__ == "__main__":
    print("🚨 AWS COST PROTECTION SETUP")
    print("=" * 50)
    
    print("\n1. Setting up billing alarms...")
    create_billing_alarms()
    
    print("\n2. Setting up monthly budget...")
    create_budget()
    
    print("\n🎯 SETUP COMPLETE!")
    print("📧 Check your email for SNS subscription confirmations")
    print("💰 You're now protected against cost disasters!")