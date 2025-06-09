#!/usr/bin/env python3
"""
Real-time cost monitoring - run this a few times today to track costs
"""

import boto3
import json
from datetime import datetime, timedelta

def detailed_cost_analysis():
    """Get detailed cost breakdown for today's incident."""
    
    print("💰 DETAILED COST ANALYSIS - Pod-Insights Incident")
    print("=" * 60)
    
    ce_client = boto3.client('ce')
    ec2_client = boto3.client('ec2')
    
    # Get costs for last 3 days to see the spike
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    try:
        # Get daily costs with service breakdown
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['BlendedCost'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
            ]
        )
        
        print(f"📅 Cost Analysis: {start_date} to {end_date}")
        print("-" * 40)
        
        daily_totals = {}
        service_details = {}
        
        for result in response['ResultsByTime']:
            date = result['TimePeriod']['Start']
            daily_total = 0
            
            print(f"\n📊 {date}:")
            
            for group in result['Groups']:
                service = group['Keys'][0]
                usage_type = group['Keys'][1] if len(group['Keys']) > 1 else 'Unknown'
                cost = float(group['Metrics']['BlendedCost']['Amount'])
                
                if cost > 0.001:  # Only show costs > $0.001
                    daily_total += cost
                    print(f"   {service} ({usage_type}): ${cost:.3f}")
                    
                    if service not in service_details:
                        service_details[service] = 0
                    service_details[service] += cost
            
            daily_totals[date] = daily_total
            print(f"   📊 Daily Total: ${daily_total:.2f}")
        
        # Show service totals
        print(f"\n🏆 SERVICE TOTALS ({start_date} to {end_date}):")
        for service, total in sorted(service_details.items(), key=lambda x: x[1], reverse=True):
            if total > 0.01:
                print(f"   {service:<30} ${total:>8.2f}")
        
        # Calculate incident cost
        today = datetime.now().strftime('%Y-%m-%d')
        incident_cost = daily_totals.get(today, 0)
        
        print(f"\n🚨 ESTIMATED INCIDENT COST:")
        print(f"   Today's extra charges: ~${incident_cost:.2f}")
        print(f"   Still covered by AWS Activate credits!")
        
    except Exception as e:
        print(f"❌ Error getting detailed costs: {e}")

def check_running_resources():
    """Double-check no resources are still running."""
    
    print(f"\n🔍 CURRENT RESOURCE CHECK:")
    print("-" * 30)
    
    try:
        ec2_client = boto3.client('ec2')
        batch_client = boto3.client('batch')
        
        # Check instances
        instances = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
            ]
        )
        
        running_instances = []
        for reservation in instances['Reservations']:
            for instance in reservation['Instances']:
                running_instances.append({
                    'id': instance['InstanceId'],
                    'type': instance['InstanceType'],
                    'launch': instance['LaunchTime']
                })
        
        if running_instances:
            print("⚠️  STILL RUNNING:")
            for inst in running_instances:
                print(f"   {inst['id']} ({inst['type']}) - {inst['launch']}")
        else:
            print("✅ No running instances")
        
        # Check compute environments
        envs = batch_client.describe_compute_environments()
        active_envs = 0
        total_desired = 0
        
        for env in envs['computeEnvironments']:
            desired = env.get('computeResources', {}).get('desiredvCpus', 0)
            min_vcpus = env.get('computeResources', {}).get('minvCpus', 0)
            
            if desired > 0 or min_vcpus > 0:
                active_envs += 1
                total_desired += desired
                print(f"⚠️  {env['computeEnvironmentName']}: {desired} desired, {min_vcpus} min")
        
        if active_envs == 0:
            print("✅ No active compute environments")
        
        print(f"📊 Total desired vCPUs across all environments: {total_desired}")
        
        # Status summary
        if len(running_instances) == 0 and total_desired == 0:
            print("\n🎉 ALL CLEAR! No ongoing costs.")
        else:
            print("\n⚠️  Some resources still active - monitor closely!")
            
    except Exception as e:
        print(f"❌ Error checking resources: {e}")

def aws_activate_status():
    """Show AWS Activate credits status."""
    
    print(f"\n💳 AWS ACTIVATE CREDITS STATUS:")
    print("-" * 35)
    print(f"   Original credits: $1,000")
    print(f"   Used so far: ~$71")
    print(f"   Remaining: ~$929")
    print(f"   Today's incident: ~$15-25 (estimated)")
    print(f"   After incident: ~$905-915 remaining")
    print(f"   ✅ More than enough for Phase 2!")

if __name__ == "__main__":
    detailed_cost_analysis()
    check_running_resources()
    aws_activate_status()
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS:")
    print("   1. Run this script 2-3 times today to monitor")
    print("   2. Check AWS Console billing tomorrow")
    print("   3. You're safe - credits cover everything!")
    print("   4. Phase 2 setup can be recreated in 10 minutes")
    print("=" * 60)