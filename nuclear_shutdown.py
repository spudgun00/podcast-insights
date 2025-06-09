#!/usr/bin/env python3
"""
Nuclear Shutdown - Absolutely ensures everything is off.
Use this when you want to guarantee zero AWS costs.
"""

import boto3
import time

def nuclear_shutdown():
    """Complete shutdown of all AWS Batch and EC2 resources."""
    
    print("☢️  NUCLEAR SHUTDOWN MODE")
    print("=" * 60)
    print("This will FORCE everything to zero cost state.")
    print("=" * 60)
    
    batch_client = boto3.client('batch')
    ec2_client = boto3.client('ec2')
    
    # 1. FORCE ALL COMPUTE ENVIRONMENTS TO EXACTLY 0 VCPUS
    print("\n☢️  STEP 1: FORCING ALL COMPUTE ENVIRONMENTS TO 0...")
    
    try:
        envs = batch_client.describe_compute_environments()
        
        for env in envs['computeEnvironments']:
            env_name = env['computeEnvironmentName']
            state = env['state']
            compute_resources = env.get('computeResources', {})
            desired_vcpus = compute_resources.get('desiredvCpus', 0)
            min_vcpus = compute_resources.get('minvCpus', 0)
            max_vcpus = compute_resources.get('maxvCpus', 0)
            
            print(f"\n📋 {env_name}:")
            print(f"   State: {state}")
            print(f"   vCPUs: min={min_vcpus}, desired={desired_vcpus}, max={max_vcpus}")
            
            # Force everything to 0
            if desired_vcpus > 0 or min_vcpus > 0:
                print(f"   🔥 FORCING TO ZERO...")
                try:
                    batch_client.update_compute_environment(
                        computeEnvironment=env_name,
                        computeResources={
                            'minvCpus': 0,
                            'desiredvCpus': 0
                        }
                    )
                    print(f"   ✅ SUCCESS: min=0, desired=0")
                except Exception as e:
                    print(f"   ❌ FAILED: {e}")
            else:
                print(f"   ✅ Already at zero")
                
    except Exception as e:
        print(f"❌ Error accessing compute environments: {e}")
    
    # 2. DISABLE ALL COMPUTE ENVIRONMENTS  
    print("\n☢️  STEP 2: DISABLING ALL COMPUTE ENVIRONMENTS...")
    
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
            else:
                print(f"✅ {env_name} already disabled")
                
    except Exception as e:
        print(f"❌ Error disabling environments: {e}")
    
    # 3. WAIT FOR BATCH TO PROCESS CHANGES
    print("\n☢️  STEP 3: WAITING FOR AWS BATCH TO PROCESS...")
    for i in range(30, 0, -1):
        print(f"   Waiting {i} seconds for AWS Batch to scale down...", end="\r")
        time.sleep(1)
    print("\n   ✅ Wait complete")
    
    # 4. TERMINATE ALL EC2 INSTANCES
    print("\n☢️  STEP 4: TERMINATING ALL EC2 INSTANCES...")
    
    try:
        # Get ALL running instances (not just GPU)
        instances = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
            ]
        )
        
        instance_ids = []
        for reservation in instances['Reservations']:
            for instance in reservation['Instances']:
                instance_id = instance['InstanceId']
                instance_type = instance['InstanceType']
                instance_ids.append(instance_id)
                print(f"🔥 Found instance: {instance_id} ({instance_type})")
        
        if instance_ids:
            print(f"\n🛑 TERMINATING {len(instance_ids)} instances...")
            ec2_client.terminate_instances(InstanceIds=instance_ids)
            print("✅ All termination commands sent")
        else:
            print("✅ No instances found to terminate")
            
    except Exception as e:
        print(f"❌ Error terminating instances: {e}")
    
    # 5. FINAL VERIFICATION
    print("\n☢️  STEP 5: FINAL VERIFICATION...")
    
    try:
        # Check instances again
        instances = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
            ]
        )
        
        running_count = sum(len(r['Instances']) for r in instances['Reservations'])
        print(f"📊 Instances still running: {running_count}")
        
        # Check compute environments
        envs = batch_client.describe_compute_environments()
        total_desired_vcpus = sum(
            env.get('computeResources', {}).get('desiredvCpus', 0) 
            for env in envs['computeEnvironments']
        )
        print(f"📊 Total desired vCPUs across all environments: {total_desired_vcpus}")
        
        if running_count == 0 and total_desired_vcpus == 0:
            print("\n🎉 NUCLEAR SHUTDOWN SUCCESSFUL!")
            print("💰 You should now have ZERO ongoing AWS compute costs.")
        else:
            print("\n⚠️  WARNING: Some resources may still be running.")
            print("Wait 5 minutes and run this script again.")
            
    except Exception as e:
        print(f"❌ Error in final verification: {e}")
    
    print("\n" + "=" * 60)
    print("☢️  NUCLEAR SHUTDOWN COMPLETE")
    print("Run 'python cost_analyzer.py' in 5 minutes to verify.")
    print("=" * 60)

if __name__ == "__main__":
    confirm = input("⚠️  This will terminate ALL running instances. Continue? (yes/no): ")
    if confirm.lower() == 'yes':
        nuclear_shutdown()
    else:
        print("Cancelled.")