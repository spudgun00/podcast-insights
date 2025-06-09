#!/usr/bin/env python3
"""
EMERGENCY AWS Shutdown Script
Forces all resources to stop when auto-scaling is out of control.
"""

import boto3
import time

def emergency_shutdown():
    """Nuclear option: force everything to stop."""
    
    print("🚨 EMERGENCY SHUTDOWN MODE")
    print("=" * 50)
    
    batch_client = boto3.client('batch')
    ec2_client = boto3.client('ec2')
    
    # 1. FORCE all compute environments to 0 vCPUs
    print("\n🛑 FORCING ALL COMPUTE ENVIRONMENTS TO 0...")
    
    try:
        envs = batch_client.describe_compute_environments()
        
        for env in envs['computeEnvironments']:
            env_name = env['computeEnvironmentName']
            desired_vcpus = env.get('computeResources', {}).get('desiredvCpus', 0)
            
            if desired_vcpus > 0:
                print(f"🔥 EMERGENCY: {env_name} has {desired_vcpus} vCPUs - FORCING TO 0")
                
                try:
                    batch_client.update_compute_environment(
                        computeEnvironment=env_name,
                        computeResources={'desiredvCpus': 0}
                    )
                    print(f"✅ Forced {env_name} to 0 vCPUs")
                except Exception as e:
                    print(f"❌ Failed to update {env_name}: {e}")
                    
    except Exception as e:
        print(f"❌ Error accessing compute environments: {e}")
    
    # 2. DISABLE all compute environments
    print("\n🛑 DISABLING ALL COMPUTE ENVIRONMENTS...")
    
    try:
        envs = batch_client.describe_compute_environments()
        
        for env in envs['computeEnvironments']:
            env_name = env['computeEnvironmentName']
            state = env['state']
            
            if state == 'ENABLED':
                print(f"🔥 DISABLING: {env_name}")
                try:
                    batch_client.update_compute_environment(
                        computeEnvironment=env_name,
                        state='DISABLED'
                    )
                    print(f"✅ Disabled {env_name}")
                except Exception as e:
                    print(f"❌ Failed to disable {env_name}: {e}")
                    
    except Exception as e:
        print(f"❌ Error disabling environments: {e}")
    
    # 3. TERMINATE all running GPU instances
    print("\n🛑 TERMINATING ALL GPU INSTANCES...")
    
    try:
        instances = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running', 'pending']},
                {'Name': 'instance-type', 'Values': ['g5.xlarge', 'g5.2xlarge', 'g5.4xlarge']}
            ]
        )
        
        instance_ids = []
        for reservation in instances['Reservations']:
            for instance in reservation['Instances']:
                instance_ids.append(instance['InstanceId'])
                print(f"🔥 Found GPU instance: {instance['InstanceId']} ({instance['InstanceType']})")
        
        if instance_ids:
            print(f"🛑 TERMINATING {len(instance_ids)} GPU instances...")
            ec2_client.terminate_instances(InstanceIds=instance_ids)
            print("✅ Termination commands sent")
        else:
            print("✅ No GPU instances found")
            
    except Exception as e:
        print(f"❌ Error terminating instances: {e}")
    
    # 4. CANCEL all batch jobs
    print("\n🛑 CANCELLING ALL BATCH JOBS...")
    
    try:
        job_queues = batch_client.describe_job_queues()
        
        for queue in job_queues['jobQueues']:
            queue_name = queue['jobQueueName']
            
            for status in ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING']:
                try:
                    jobs = batch_client.list_jobs(
                        jobQueue=queue_name,
                        jobStatus=status
                    )
                    
                    for job in jobs.get('jobList', []):
                        job_id = job['jobId']
                        print(f"🛑 Cancelling job: {job_id}")
                        batch_client.cancel_job(
                            jobId=job_id,
                            reason='Emergency shutdown'
                        )
                        
                except Exception as e:
                    print(f"❌ Error cancelling jobs in {queue_name}: {e}")
                    
    except Exception as e:
        print(f"❌ Error accessing job queues: {e}")
    
    print("\n" + "=" * 50)
    print("🚨 EMERGENCY SHUTDOWN COMPLETE")
    print("Wait 2-3 minutes, then run:")
    print("python cost_analyzer.py")
    print("to verify everything is stopped.")

if __name__ == "__main__":
    emergency_shutdown()