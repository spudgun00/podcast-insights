#!/usr/bin/env python3
import subprocess
import click
import os
from pathlib import Path
import sys

# Determine the project root directory (assuming podrun.py is in cli/ relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@click.group()
def cli():
    """Pod-Insight CLI entry point (podrun)"""
    pass

@cli.command()
@click.option("--limit", type=int, default=0, help="Limit the number of episodes to process from feeds for manifest generation, and subsequently for fetching.")
@click.option("--dry-run", is_flag=True, help="Perform a dry run; generate manifest and show backfill command, but don't execute backfill.")
@click.option("--config-file", default="tier1_feeds.yaml", help="Path to the feed configuration YAML file for manifest generation.")
@click.option("--manifest-output-name", default="manifest.csv", help="Name of the manifest file to be generated locally.")
# In a cloud environment, generate_manifest.py would upload to S3 and output an S3 URI.
# For local podrun, we'll have generate_manifest.py create a local file that backfill.py then uses.
def fetch(limit, dry_run, config_file, manifest_output_name):
    """Generates a manifest and then fetches audio and minimal metadata.

    1. Runs tools/generate_manifest.py to create a manifest.

    2. Calls backfill.py --mode fetch using the generated manifest.
    """
    click.echo("Step 1: Generating manifest...")
    
    generate_manifest_script = PROJECT_ROOT / "tools" / "generate_manifest.py"
    # Adjust generate_manifest.py to accept output path and config path
    # For now, assume it creates manifest.csv in its directory or CWD
    # and can take --config and --output-csv arguments
    
    # We want the manifest to be created in a predictable place, e.g., PROJECT_ROOT
    local_manifest_path = PROJECT_ROOT / manifest_output_name

    cmd_generate_manifest = [
        "python", str(generate_manifest_script),
        "--config", str(PROJECT_ROOT / config_file), # Assuming generate_manifest takes this
        "--output-csv", str(local_manifest_path),    # Assuming generate_manifest takes this
        # generate_manifest.py might also take --since, but we'll let its default or config handle it
    ]
    if limit > 0: # If generate_manifest.py supports a limit for number of feed items
        # cmd_generate_manifest.extend(["--limit", str(limit)]) # generate_manifest.py doesn't have a limit currently
        pass # generate_manifest.py currently processes all items from feeds based on SINCE_DATE

    click.echo(f"Running: {' '.join(cmd_generate_manifest)}")
    try:
        # Assuming generate_manifest.py prints the path of the generated manifest if successful
        # or we use a fixed local path as done with local_manifest_path
        process_gen_manifest = subprocess.run(cmd_generate_manifest, check=True, capture_output=True, text=True)
        click.echo(f"Manifest generation successful. Output:")
        click.echo(process_gen_manifest.stdout)
        if process_gen_manifest.stderr:
            click.echo("Stderr:")
            click.echo(process_gen_manifest.stderr)

        # Check if manifest file was actually created
        if not local_manifest_path.exists():
            click.secho(f"Error: Manifest file {local_manifest_path} was not created by generate_manifest.py.", fg="red")
            return 1
        
        manifest_to_use = str(local_manifest_path) # Use the locally generated manifest

    except subprocess.CalledProcessError as e:
        click.secho(f"Manifest generation failed with exit code {e.returncode}:", fg="red")
        click.secho(e.stdout, fg="red")
        click.secho(e.stderr, fg="red")
        return 1
    except FileNotFoundError:
        click.secho(f"Error: generate_manifest.py script not found at {generate_manifest_script}", fg="red")
        return 1


    click.echo("\nStep 2: Calling backfill.py --mode fetch...")
    backfill_script = PROJECT_ROOT / "backfill.py"
    backfill_cmd = [
        "python", str(backfill_script),
        "--mode", "fetch",
        "--manifest", manifest_to_use, # Pass the path to the generated manifest
    ]
    if limit > 0: # backfill.py --mode fetch supports --limit
        backfill_cmd.extend(["--limit", str(limit)])
    if dry_run: # backfill.py supports --dry_run
        backfill_cmd.append("--dry_run")

    click.echo(f"Running: {' '.join(backfill_cmd)}")
    if dry_run and not limit: # If it's just a dry run of showing the command structure
        click.echo("Dry run: backfill.py command prepared. Not executing.")
        return 0
    
    try:
        # For long running processes, consider streaming output or just running without capture
        subprocess.run(backfill_cmd, check=True) 
        click.secho("backfill.py fetch completed successfully.", fg="green")
    except subprocess.CalledProcessError as e:
        click.secho(f"backfill.py fetch failed with exit code {e.returncode}.", fg="red")
        return 1
    except FileNotFoundError:
        click.secho(f"Error: backfill.py script not found at {backfill_script}", fg="red")
        return 1
    return 0

@cli.command()
@click.option("--limit", type=int, default=0, help="Limit the number of episodes to process from the manifest.")
@click.option("--manifest", required=True, help="Path or S3 URI to the manifest CSV file. (e.g., manifest.csv or s3://bucket/manifest.csv)")
@click.option("--model", default=None, help="Whisper model size (tiny, base, small, medium, large). Overrides backfill.py config.")
@click.option("--offset", type=int, default=0, help="Skip this many episodes from manifest start (for array job processing).")
@click.option("--dry-run", is_flag=True, help="Perform a dry run; show backfill command but don't execute.")
def transcribe(limit, manifest, model, offset, dry_run):
    """Transcribes audio based on a manifest, using backfill.py --mode transcribe.

    Requires a manifest file (local or S3 URI) typically generated by the 'fetch' step 

    or a prior run of 'generate_manifest.py'.
    """
    click.echo("Calling backfill.py --mode transcribe...")
    backfill_script = PROJECT_ROOT / "backfill.py"

    backfill_cmd = [
        sys.executable, str(backfill_script),
        "--mode", "transcribe",
        "--manifest", manifest, # User provides the manifest path directly
    ]
    if limit > 0:
        backfill_cmd.extend(["--limit", str(limit)])
    if model: # backfill.py takes --model_size
        backfill_cmd.extend(["--model_size", model])
    if offset > 0:
        backfill_cmd.extend(["--offset", str(offset)])
    if dry_run:
        backfill_cmd.append("--dry_run")

    click.echo(f"Running: {' '.join(backfill_cmd)}")
    if dry_run and not limit and not model: # If it's just a dry run of showing the command structure
        click.echo("Dry run: backfill.py command prepared. Not executing.")
        return 0

    try:
        # Explicitly pass relevant AWS environment variables
        env = os.environ.copy()
        # The profile and region should be set in the shell running podrun.py
        # backfill.py will then pick them up via its own os.getenv calls.
        # We are just ensuring they are part of the environment passed to the subprocess.
        # No need to set them if they are already correctly in os.environ from the parent shell.
        # However, to be absolutely sure they are passed if set:
        if "AWS_PROFILE" in os.environ:
            env["AWS_PROFILE"] = os.environ["AWS_PROFILE"]
        if "AWS_REGION" in os.environ:
            env["AWS_REGION"] = os.environ["AWS_REGION"]
        if "NO_AWS" in os.environ: # Also pass NO_AWS if it was set/unset
            env["NO_AWS"] = os.environ["NO_AWS"]
        else: # If unset NO_AWS was used, make sure it's not in env for subprocess
            if "NO_AWS" in env: del env["NO_AWS"]

        # Debugging: Print AWS-related environment variables
        click.echo("--- Debugging AWS Env Vars in podrun.py ---")
        click.echo(f"os.environ.get('AWS_PROFILE'): {os.environ.get('AWS_PROFILE')}")
        click.echo(f"os.environ.get('AWS_REGION'): {os.environ.get('AWS_REGION')}")
        click.echo(f"os.environ.get('AWS_ACCESS_KEY_ID') set: {'AWS_ACCESS_KEY_ID' in os.environ}")
        click.echo(f"env.get('AWS_PROFILE'): {env.get('AWS_PROFILE')}")
        click.echo(f"env.get('AWS_REGION'): {env.get('AWS_REGION')}")
        click.echo(f"env.get('AWS_ACCESS_KEY_ID') set: {'AWS_ACCESS_KEY_ID' in env}")
        click.echo("-------------------------------------------")

        subprocess.run(backfill_cmd, check=True, env=env)
        click.secho("backfill.py transcribe completed successfully.", fg="green")
    except subprocess.CalledProcessError as e:
        click.secho(f"backfill.py transcribe failed with exit code {e.returncode}.", fg="red")
        return 1
    except FileNotFoundError:
        click.secho(f"Error: backfill.py script not found at {backfill_script}", fg="red")
        return 1
    return 0

if __name__ == '__main__':
    cli() 