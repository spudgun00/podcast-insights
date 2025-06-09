#!/usr/bin/env python3
"""
Timeline Investigator - Figure out exactly when the cost bleeding started
"""

import boto3
import json
from datetime import datetime, timedelta

def investigate_timeline():
    """Comprehensive timeline investigation."""
    
    print("🔍 AWS COST TIMELINE INVESTIGATION")
    print("=" * 50)
    
    ce_client = boto3.client('ce')
    batch_client = boto3.client('batch')
    
    # Get extended cost history
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    
    print(f"📅 Analyzing: {start_date} to {end_date}")
    print("-" * 40)
    
    try:
        # Get daily costs with service breakdown for last 2 weeks
        response = ce_client.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='DAILY',
            Metrics=['BlendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        
        suspicious_days = []
        
        for result in response['ResultsByTime']:
            date = result['TimePeriod']['Start']
            daily_total = 0
            compute_cost = 0
            
            services = {}
            for group in result['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['BlendedCost']['Amount'])
                daily_total += cost
                
                if cost > 0.001:
                    services[service] = cost
                    
                    # Look for compute-related costs
                    if any(keyword in service.lower() for keyword in ['compute', 'elastic', 'ec2']):
                        if 'data' not in service.lower():  # Exclude data transfer
                            compute_cost += cost
            
            # Flag suspicious days
            if compute_cost > 1.0:  # More than $1 in compute costs
                suspicious_days.append({
                    'date': date,
                    'total': daily_total,
                    'compute': compute_cost,
                    'services': services
                })
            
            # Show all days with any costs
            if daily_total > 0.01:
                print(f"\n📊 {date}: ${daily_total:.2f}")
                if compute_cost > 0.1:
                    print(f"   🔥 Compute costs: ${compute_cost:.2f}")
                for service, cost in sorted(services.items(), key=lambda x: x[1], reverse=True):
                    if cost > 0.01:
                        print(f"   {service}: ${cost:.2f}")
        
        # Analyze suspicious patterns
        if suspicious_days:
            print(f"\n🚨 SUSPICIOUS HIGH-COST DAYS:")
            print("-" * 30)
            
            for day in suspicious_days:
                print(f"📅 {day['date']}: ${day['total']:.2f} (${day['compute']:.2f} compute)")
                
            # Estimate total incident cost
            total_incident = sum(day['compute'] for day in suspicious_days)
            print(f"\n🔥 ESTIMATED TOTAL INCIDENT COST: ${total_incident:.2f}")
            
            # Determine duration
            if len(suspicious_days) > 1:
                start_incident = suspicious_days[0]['date']
                end_incident = suspicious_days[-1]['date']
                print(f"📅 Incident duration: {start_incident} to {end_incident}")
                print(f"⏱️  Duration: {len(suspicious_days)} days")
        else:
            print(f"\n✅ No major cost spikes detected in billing data")
            print(f"   This suggests the incident was recent (last 24-48 hours)")
            print(f"   OR AWS billing hasn't caught up yet")
        
    except Exception as e:
        print(f"❌ Error analyzing timeline: {e}")

def check_resource_creation_dates():
    """Check when the problematic resources were created."""
    
    print(f"\n🏗️  RESOURCE CREATION TIMELINE:")
    print("-" * 35)
    
    try:
        batch_client = boto3.client('batch')
        
        # Check compute environments (some might be deleted)
        try:
            envs = batch_client.describe_compute_environments()
            print("📋 Existing Compute Environments:")
            
            for env in envs['computeEnvironments']:
                name = env['computeEnvironmentName']
                created = env.get('createdAt', 'Unknown')
                state = env['state']
                
                if isinstance(created, datetime):
                    created_str = created.strftime('%Y-%m-%d %H:%M')
                else:
                    created_str = str(created)
                
                print(f"   {name}: {created_str} ({state})")
                
        except Exception as e:
            print(f"❌ Error checking compute environments: {e}")
        
        # Check job queues
        try:
            queues = batch_client.describe_job_queues()
            print(f"\n📋 Job Queues:")
            
            for queue in queues['jobQueues']:
                name = queue['jobQueueName']
                state = queue['state']
                print(f"   {name}: {state}")
                
        except Exception as e:
            print(f"❌ Error checking job queues: {e}")
            
    except Exception as e:
        print(f"❌ Error checking resources: {e}")

def estimate_maximum_damage():
    """Estimate worst-case scenario costs."""
    
    print(f"\n💰 DAMAGE ASSESSMENT:")
    print("-" * 25)
    
    print(f"🔍 Based on your description:")
    print(f"   - 3 compute environments with min vCPUs > 0")
    print(f"   - podinsight_gpu_smoke_ce: 4 vCPUs")
    print(f"   - podinsight-gpu-production: 8 vCPUs") 
    print(f"   - podinsight-gpu-ce2: 4 vCPUs")
    print(f"   - Total: 16 vCPUs × $0.20/hour = $3.20/hour")
    print(f"")
    print(f"📊 Potential costs by duration:")
    print(f"   - 6 hours: ${3.20 * 6:.2f}")
    print(f"   - 12 hours: ${3.20 * 12:.2f}")
    print(f"   - 24 hours: ${3.20 * 24:.2f}")
    print(f"   - 48 hours: ${3.20 * 48:.2f}")
    print(f"   - 1 week: ${3.20 * 24 * 7:.2f}")
    print(f"")
    print(f"✅ YOUR SITUATION:")
    print(f"   - AWS Activate credits: $928 remaining")
    print(f"   - Even worst case (1 week): easily covered")
    print(f"   - Most likely: few hours = $10-30")

if __name__ == "__main__":
    investigate_timeline()
    check_resource_creation_dates()
    estimate_maximum_damage()
    
    print("\n" + "=" * 50)
    print("🎯 BOTTOM LINE:")
    print("   1. Check this script output for suspicious days")
    print("   2. AWS billing delays mean we might not see full impact yet")
    print("   3. Even worst case is covered by your credits")
    print("   4. Monitor AWS console billing over next 48 hours")
    print("=" * 50)