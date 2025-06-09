#!/usr/bin/env python3
"""
AWS Cost Analysis Script for Pod-Insights Project
Helps identify what's driving your costs and what resources are running.
"""

import boto3
import json
from datetime import datetime, timedelta
import pandas as pd

def analyze_current_costs():
    """Analyze costs for the current month to identify expensive services."""
    
    # Initialize AWS clients
    ce_client = boto3.client('ce')  # Cost Explorer
    ec2_client = boto3.client('ec2')
    batch_client = boto3.client('batch')
    
    print("🔍 ANALYZING YOUR AWS COSTS...")
    print("=" * 50)
    
    # Get cost data for current month
    start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['BlendedCost'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ]
        )
        
        print(f"📅 Cost Analysis: {start_date} to {end_date}")
        print("-" * 30)
        
        # Parse and display costs by service
        service_costs = {}
        for result in response['ResultsByTime']:
            date = result['TimePeriod']['Start']
            for group in result['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['BlendedCost']['Amount'])
                if service not in service_costs:
                    service_costs[service] = 0
                service_costs[service] += cost
        
        # Sort by cost (highest first)
        sorted_costs = sorted(service_costs.items(), key=lambda x: x[1], reverse=True)
        
        total_cost = sum(service_costs.values())
        print(f"💰 Total Month-to-Date Cost: ${total_cost:.2f}")
        print("\n🏆 Top Cost Drivers:")
        
        for service, cost in sorted_costs[:10]:
            percentage = (cost / total_cost) * 100 if total_cost > 0 else 0
            print(f"  {service:<30} ${cost:>8.2f} ({percentage:>5.1f}%)")
            
    except Exception as e:
        print(f"❌ Error getting cost data: {e}")
        print("   Make sure you have cost explorer permissions")

def check_running_resources():
    """Check what EC2 and Batch resources are currently running."""
    
    print("\n🖥️  CHECKING RUNNING RESOURCES...")
    print("=" * 50)
    
    try:
        ec2_client = boto3.client('ec2')
        batch_client = boto3.client('batch')
        
        # Check EC2 instances
        print("🔍 EC2 Instances:")
        instances = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
            ]
        )
        
        running_instances = []
        for reservation in instances['Reservations']:
            for instance in reservation['Instances']:
                running_instances.append({
                    'InstanceId': instance['InstanceId'],
                    'InstanceType': instance['InstanceType'],
                    'State': instance['State']['Name'],
                    'LaunchTime': instance['LaunchTime']
                })
        
        if running_instances:
            for inst in running_instances:
                runtime = datetime.now(inst['LaunchTime'].tzinfo) - inst['LaunchTime']
                print(f"  ⚠️  {inst['InstanceId']} ({inst['InstanceType']}) - Running {runtime.days}d {runtime.seconds//3600}h")
        else:
            print("  ✅ No running EC2 instances")
        
        # Check Batch Compute Environments
        print("\n🔍 Batch Compute Environments:")
        envs = batch_client.describe_compute_environments()
        
        active_envs = []
        for env in envs['computeEnvironments']:
            if env['state'] == 'ENABLED' and env['status'] == 'VALID':
                active_envs.append(env)
        
        if active_envs:
            for env in active_envs:
                desired_vcpus = env['computeResources'].get('desiredvCpus', 0)
                max_vcpus = env['computeResources'].get('maxvCpus', 0)
                print(f"  ⚠️  {env['computeEnvironmentName']}")
                print(f"      Desired vCPUs: {desired_vcpus}, Max: {max_vcpus}")
                if desired_vcpus > 0:
                    print(f"      🚨 COSTING MONEY - {desired_vcpus} vCPUs running!")
        else:
            print("  ✅ No active compute environments")
            
        # Check for running Batch jobs
        print("\n🔍 Active Batch Jobs:")
        job_queues = batch_client.describe_job_queues()
        
        active_jobs = 0
        for queue in job_queues['jobQueues']:
            if queue['state'] == 'ENABLED':
                jobs = batch_client.list_jobs(
                    jobQueue=queue['jobQueueName'],
                    jobStatus='RUNNING'
                )
                # FIX: Changed from 'jobList' to 'jobSummaryList'
                if jobs['jobSummaryList']:
                    active_jobs += len(jobs['jobSummaryList'])
                    print(f"  ⚠️  {len(jobs['jobSummaryList'])} jobs running in {queue['jobQueueName']}")
        
        if active_jobs == 0:
            print("  ✅ No running batch jobs")
            
    except Exception as e:
        print(f"❌ Error checking resources: {e}")

def find_recent_successful_jobs():
    """Find recent successful batch jobs to correlate with S3 uploads."""
    
    print("\n📋 RECENT BATCH JOB HISTORY...")
    print("=" * 50)
    
    try:
        batch_client = boto3.client('batch')
        
        # Get job queues
        job_queues = batch_client.describe_job_queues()
        
        print("🔍 Searching for jobs from May 28-30 (around your S3 upload date)...")
        
        for queue in job_queues['jobQueues']:
            queue_name = queue['jobQueueName']
            print(f"\n📁 Queue: {queue_name}")
            
            # Check different job statuses
            for status in ['SUCCEEDED', 'FAILED', 'RUNNING']:
                try:
                    jobs = batch_client.list_jobs(
                        jobQueue=queue_name,
                        jobStatus=status,
                        maxResults=20
                    )
                    
                    # FIX: Changed from 'jobList' to 'jobSummaryList'
                    if jobs['jobSummaryList']:
                        print(f"  {status} jobs:")
                        for job in jobs['jobSummaryList'][:5]:  # Show first 5
                            job_detail = batch_client.describe_jobs(jobs=[job['jobId']])
                            job_info = job_detail['jobs'][0]
                            
                            start_time = job_info.get('startedAt', 'Unknown')
                            if isinstance(start_time, int):
                                start_time = datetime.fromtimestamp(start_time/1000)
                            
                            print(f"    {job['jobName']} - {start_time}")
                    else:
                        print(f"    No {status} jobs found")
                            
                except Exception as inner_e:
                    print(f"    Error checking {status} jobs: {inner_e}")
                    
    except Exception as e:
        print(f"❌ Error finding jobs: {e}")

def generate_cost_optimization_report():
    """Generate recommendations for cost optimization."""
    
    print("\n💡 COST OPTIMIZATION RECOMMENDATIONS...")
    print("=" * 50)
    
    recommendations = [
        "🎯 IMMEDIATE ACTIONS:",
        "  1. Set Batch Compute Environment desired vCPUs to 0 when not in use",
        "  2. Terminate any idle EC2 instances immediately",
        "  3. Use Spot instances for all non-critical workloads",
        "",
        "🔄 WORKFLOW BEST PRACTICES:",
        "  1. Scale compute environments to 0 after each job completes",
        "  2. Use AWS Lambda to auto-shutdown resources after jobs",
        "  3. Set up CloudWatch alarms for cost thresholds",
        "",
        "📊 MONITORING SETUP:",
        "  1. Create daily cost budgets with alerts",
        "  2. Tag all resources with project names",
        "  3. Use Cost Explorer daily during development",
        "",
        "⏰ FOR YOUR POD-INSIGHTS PROJECT:",
        "  1. Only enable compute environments when running jobs",
        "  2. Use smaller instance types for testing (t3.micro vs g5.xlarge)",
        "  3. Run Phase 2 in smaller batches (10-20 episodes at a time)",
        "  4. Monitor costs before each major run"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    print("🚀 AWS COST ANALYSIS FOR POD-INSIGHTS PROJECT")
    print("=" * 60)
    
    try:
        analyze_current_costs()
        check_running_resources()
        find_recent_successful_jobs()
        generate_cost_optimization_report()
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete! Review the recommendations above.")
        print("💰 Remember: Your costs are covered by AWS Activate credits.")
        print("🎯 Goal: Keep future runs under $35 as planned.")
        
    except Exception as e:
        print(f"❌ Script error: {e}")
        print("Make sure you have AWS credentials configured and appropriate permissions.")