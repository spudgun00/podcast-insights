#!/usr/bin/env python3
"""
AWS Batch Complete Cleanup - Handles the proper order of operations
"""

import boto3
import time

def complete_batch_cleanup():
    """Properly cleanup AWS Batch resources in the correct order."""
    
    print("🧹 AWS BATCH COMPLETE CLEANUP")
    print("=" * 50)
    
    batch_client = boto3.client('batch')
    ec2_client = boto3.client('ec2')
    
    # Step 1: Kill any remaining instances immediately
    print("🔥 STEP 1: Terminate remaining instances...")
    try:
        instances = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
            ]
        )
        
        instance_ids = []
        for reservation in instances['Reservations']:
            for instance in reservation['Instances']:
                instance_ids.append(instance['InstanceId'])
                print(f"🔥 Found: {instance['InstanceId']} ({instance['InstanceType']})")
        
        if instance_ids:
            ec2_client.terminate_instances(InstanceIds=instance_ids)
            print(f"✅ Terminated {len(instance_ids)} instances")
        else:
            print("✅ No instances to terminate")
    except Exception as e:
        print(f"❌ Error terminating instances: {e}")
    
    # Step 2: Get all job queues and their relationships
    print("\n🔍 STEP 2: Analyzing job queue relationships...")
    try:
        job_queues = batch_client.describe_job_queues()
        
        queue_relationships = {}
        for queue in job_queues['jobQueues']:
            queue_name = queue['jobQueueName']
            state = queue['state']
            compute_envs = [ce['computeEnvironment'] for ce in queue.get('computeEnvironmentOrder', [])]
            
            print(f"📋 Queue: {queue_name} (State: {state})")
            for ce in compute_envs:
                print(f"   🔗 Connected to: {ce}")
                if ce not in queue_relationships:
                    queue_relationships[ce] = []
                queue_relationships[ce].append(queue_name)
        
        return queue_relationships
        
    except Exception as e:
        print(f"❌ Error analyzing relationships: {e}")
        return {}

def disconnect_all_queues(queue_relationships):
    """Disconnect all job queues from compute environments."""
    
    print("\n🔌 STEP 3: Disconnecting job queues from compute environments...")
    batch_client = boto3.client('batch')
    
    try:
        job_queues = batch_client.describe_job_queues()
        
        for queue in job_queues['jobQueues']:
            queue_name = queue['jobQueueName']
            
            if queue.get('computeEnvironmentOrder'):
                print(f"🔌 Disconnecting queue: {queue_name}")
                try:
                    batch_client.update_job_queue(
                        jobQueue=queue_name,
                        state='DISABLED',
                        computeEnvironmentOrder=[]  # Empty list disconnects all
                    )
                    print(f"✅ Disconnected {queue_name}")
                except Exception as e:
                    print(f"❌ Failed to disconnect {queue_name}: {e}")
            else:
                print(f"✅ {queue_name} already disconnected")
                
    except Exception as e:
        print(f"❌ Error disconnecting queues: {e}")

def delete_problematic_environments():
    """Delete the compute environments with min vCPUs > 0."""
    
    print("\n🗑️  STEP 4: Deleting problematic compute environments...")
    batch_client = boto3.client('batch')
    
    problem_envs = [
        'podinsight_gpu_smoke_ce',
        'podinsight-gpu-production', 
        'podinsight-gpu-ce2'
    ]
    
    for env_name in problem_envs:
        print(f"🗑️  Attempting to delete: {env_name}")
        try:
            batch_client.delete_compute_environment(computeEnvironment=env_name)
            print(f"✅ Deletion initiated for {env_name}")
        except Exception as e:
            if "does not exist" in str(e):
                print(f"✅ {env_name} already deleted")
            else:
                print(f"❌ Failed to delete {env_name}: {e}")

def wait_and_verify():
    """Wait for AWS changes to propagate and verify cleanup."""
    
    print("\n⏳ STEP 5: Waiting for AWS to process changes...")
    for i in range(60, 0, -1):
        print(f"   Waiting {i} seconds...", end="\r")
        time.sleep(1)
    
    print("\n🔍 STEP 6: Final verification...")
    
    batch_client = boto3.client('batch')
    ec2_client = boto3.client('ec2')
    
    # Check instances
    try:
        instances = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
            ]
        )
        running_count = sum(len(r['Instances']) for r in instances['Reservations'])
        print(f"📊 Running instances: {running_count}")
    except Exception as e:
        print(f"❌ Error checking instances: {e}")
    
    # Check compute environments
    try:
        envs = batch_client.describe_compute_environments()
        total_desired = 0
        problematic_envs = 0
        
        for env in envs['computeEnvironments']:
            desired = env.get('computeResources', {}).get('desiredvCpus', 0)
            min_vcpus = env.get('computeResources', {}).get('minvCpus', 0)
            total_desired += desired
            
            if min_vcpus > 0:
                problematic_envs += 1
                print(f"⚠️  {env['computeEnvironmentName']}: min={min_vcpus} (PROBLEMATIC)")
        
        print(f"📊 Total desired vCPUs: {total_desired}")
        print(f"📊 Environments with min > 0: {problematic_envs}")
        
        if running_count == 0 and total_desired == 0 and problematic_envs == 0:
            print("\n🎉 COMPLETE SUCCESS!")
            print("💰 All problematic resources cleaned up!")
        else:
            print("\n⚠️  Some issues remain. You may need to:")
            print("   1. Wait longer for AWS to process")
            print("   2. Manually check AWS Console")
            print("   3. Delete environments via console")
            
    except Exception as e:
        print(f"❌ Error checking environments: {e}")

if __name__ == "__main__":
    # Run complete cleanup
    queue_relationships = complete_batch_cleanup()
    disconnect_all_queues(queue_relationships)
    
    print("\n⏳ Waiting 30 seconds for disconnections to process...")
    time.sleep(30)
    
    delete_problematic_environments()
    wait_and_verify()
    
    print("\n" + "=" * 50)
    print("🧹 CLEANUP COMPLETE")
    print("Run 'python cost_analyzer.py' to verify.")
    print("=" * 50)
    