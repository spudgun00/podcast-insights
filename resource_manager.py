#!/usr/bin/env python3
"""
AWS Resource Management Script for Pod-Insights Project
Helps you properly start/stop/scale resources to control costs.
"""

import boto3
import json
import time
from datetime import datetime

def scale_batch_environments(action="status"):
    """
    Manage Batch Compute Environments - the biggest cost driver.
    Actions: 'stop', 'start', 'status'
    """
    
    print(f"🔧 BATCH COMPUTE ENVIRONMENT MANAGEMENT - {action.upper()}")
    print("=" * 60)
    
    try:
        batch_client = boto3.client('batch')
        
        # Get all compute environments
        envs = batch_client.describe_compute_environments()
        
        for env in envs['computeEnvironments']:
            env_name = env['computeEnvironmentName']
            current_state = env['state']
            current_status = env['status']
            
            compute_resources = env.get('computeResources', {})
            desired_vcpus = compute_resources.get('desiredvCpus', 0)
            max_vcpus = compute_resources.get('maxvCpus', 0)
            
            print(f"\n📋 Environment: {env_name}")
            print(f"   State: {current_state}, Status: {current_status}")
            print(f"   Current vCPUs: {desired_vcpus}/{max_vcpus}")
            
            if action == "stop":
                if desired_vcpus > 0:
                    print(f"   🛑 Scaling down to 0 vCPUs...")
                    batch_client.update_compute_environment(
                        computeEnvironment=env_name,
                        computeResources={
                            'desiredvCpus': 0
                        }
                    )
                    print(f"   ✅ Scaled down - will save ~${desired_vcpus * 0.05:.2f}/hour")
                else:
                    print(f"   ✅ Already at 0 vCPUs")
                    
            elif action == "start":
                if desired_vcpus == 0 and max_vcpus > 0:
                    # Scale up to 25% of max for moderate usage
                    target_vcpus = max(1, max_vcpus // 4)
                    print(f"   🚀 Scaling up to {target_vcpus} vCPUs...")
                    batch_client.update_compute_environment(
                        computeEnvironment=env_name,
                        computeResources={
                            'desiredvCpus': target_vcpus
                        }
                    )
                    print(f"   ✅ Scaled up - will cost ~${target_vcpus * 0.05:.2f}/hour")
                else:
                    print(f"   ℹ️  Already running or max_vcpus is 0")
            
            # Show cost impact
            if desired_vcpus > 0:
                hourly_cost = desired_vcpus * 0.05  # Rough estimate
                daily_cost = hourly_cost * 24
                print(f"   💰 Current cost: ~${hourly_cost:.2f}/hour, ${daily_cost:.2f}/day")
                
    except Exception as e:
        print(f"❌ Error managing batch environments: {e}")

def terminate_idle_instances():
    """Find and optionally terminate idle EC2 instances."""
    
    print("\n🖥️  EC2 INSTANCE MANAGEMENT")
    print("=" * 40)
    
    try:
        ec2_client = boto3.client('ec2')
        
        # Get running instances
        instances = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running']}
            ]
        )
        
        idle_instances = []
        
        for reservation in instances['Reservations']:
            for instance in reservation['Instances']:
                instance_id = instance['InstanceId']
                instance_type = instance['InstanceType']
                launch_time = instance['LaunchTime']
                
                # Calculate runtime
                runtime = datetime.now(launch_time.tzinfo) - launch_time
                runtime_hours = runtime.total_seconds() / 3600
                
                print(f"\n📋 Instance: {instance_id} ({instance_type})")
                print(f"   Runtime: {runtime.days}d {runtime.seconds//3600}h")
                print(f"   Launched: {launch_time}")
                
                # Check if it might be idle (running > 1 hour)
                if runtime_hours > 1:
                    idle_instances.append({
                        'id': instance_id,
                        'type': instance_type,
                        'runtime_hours': runtime_hours
                    })
                    print(f"   ⚠️  Potentially idle - consider terminating")
                    
                    # Rough cost calculation
                    cost_per_hour = {
                        't3.micro': 0.0104, 't3.small': 0.0208, 't3.medium': 0.0416,
                        'g5.xlarge': 1.006, 'g5.2xlarge': 2.012
                    }
                    hourly_rate = cost_per_hour.get(instance_type, 0.10)
                    total_cost = runtime_hours * hourly_rate
                    print(f"   💰 Estimated cost so far: ${total_cost:.2f}")
        
        if idle_instances:
            print(f"\n⚠️  Found {len(idle_instances)} potentially idle instances")
            print("   💡 Consider terminating them if not needed")
            print("   🔧 Use: aws ec2 terminate-instances --instance-ids <id>")
        else:
            print("\n✅ No long-running instances found")
            
    except Exception as e:
        print(f"❌ Error checking instances: {e}")

def setup_cost_monitoring():
    """Set up cost monitoring and budgets."""
    
    print("\n📊 COST MONITORING SETUP")
    print("=" * 30)
    
    try:
        budgets_client = boto3.client('budgets')
        
        # Create a simple budget for the pod-insights project
        budget_name = "pod-insights-monthly-budget"
        budget_limit = 50.0  # $50/month budget
        
        budget = {
            'BudgetName': budget_name,
            'BudgetLimit': {
                'Amount': str(budget_limit),
                'Unit': 'USD'
            },
            'TimeUnit': 'MONTHLY',
            'BudgetType': 'COST',
            'CostFilters': {
                # You can add specific service filters here
            }
        }
        
        # Create notification for 80% threshold
        notification = {
            'NotificationType': 'ACTUAL',
            'ComparisonOperator': 'GREATER_THAN',
            'Threshold': 80.0,  # 80% of budget
            'ThresholdType': 'PERCENTAGE'
        }
        
        print(f"💡 RECOMMENDED BUDGET SETUP:")
        print(f"   Budget Name: {budget_name}")
        print(f"   Monthly Limit: ${budget_limit}")
        print(f"   Alert at: {notification['Threshold']}% (${budget_limit * 0.8})")
        print(f"   \n   To create this budget, use AWS Console > Billing > Budgets")
        print(f"   or run: aws budgets create-budget --account-id YOUR_ACCOUNT_ID")
        
    except Exception as e:
        print(f"❌ Error setting up monitoring: {e}")

def generate_shutdown_checklist():
    """Generate a checklist for shutting down resources after work."""
    
    checklist = [
        "🔄 END-OF-DAY SHUTDOWN CHECKLIST:",
        "=" * 40,
        "",
        "📋 BEFORE LOGGING OFF:",
        "  □ Scale all Batch Compute Environments to 0 vCPUs",
        "  □ Terminate any running EC2 instances (if not needed overnight)",
        "  □ Check that no Batch jobs are queued/running",
        "  □ Verify CloudWatch shows decreasing costs",
        "",
        "📋 BEFORE MAJOR RUNS:",
        "  □ Check current AWS costs in billing dashboard",
        "  □ Estimate cost of planned run (episodes × $0.15)",
        "  □ Scale up only the needed compute environments",
        "  □ Monitor job progress and costs",
        "",
        "📋 WEEKLY REVIEWS:",
        "  □ Review total costs vs. AWS Activate credits remaining",
        "  □ Identify any unexpected cost spikes",
        "  □ Cleanup any leftover resources",
        "  □ Update team on cost efficiency",
        "",
        "🚨 EMERGENCY STOPS:",
        "  □ If costs spike unexpectedly, immediately scale all environments to 0",
        "  □ Use this script: python resource_manager.py --action stop",
        "  □ Check for runaway batch jobs or instances",
    ]
    
    print("\n" + "\n".join(checklist))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage Pod-Insights AWS Resources')
    parser.add_argument('--action', choices=['status', 'stop', 'start', 'monitor', 'checklist'], 
                       default='status', help='Action to perform')
    
    args = parser.parse_args()
    
    print("🛠️  POD-INSIGHTS RESOURCE MANAGER")
    print("=" * 50)
    
    if args.action in ['status', 'stop', 'start']:
        scale_batch_environments(args.action)
        terminate_idle_instances()
        
    elif args.action == 'monitor':
        setup_cost_monitoring()
        
    elif args.action == 'checklist':
        generate_shutdown_checklist()
    
    print(f"\n💡 QUICK COMMANDS:")
    print(f"   Stop all:     python resource_manager.py --action stop")
    print(f"   Start modest: python resource_manager.py --action start") 
    print(f"   Check status: python resource_manager.py --action status")
    print(f"   Get checklist:python resource_manager.py --action checklist")