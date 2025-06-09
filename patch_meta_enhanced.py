#!/usr/bin/env python3
"""
Enhanced metadata patch script that does better extraction from filenames
"""
import json, glob, os, re

def clean_name(text):
    """Convert filenames to more readable podcast names"""
    # Replace special characters and hyphens with spaces
    text = re.sub(r'[-_]', ' ', text)
    
    # Special case handling
    if "Equity" in text:
        return "TechCrunch Equity"
    elif "How I Built This" in text:
        return "How I Built This with Guy Raz"
    elif "24d48b2f" in text:
        return "Substack Podcast"
    elif "Fridman" in text:
        return "Lex Fridman Podcast"
    elif "First Million" in text:
        return "My First Million"
    elif "Twenty Minute" in text or "20VC" in text:
        return "The Twenty Minute VC"
    elif "Pitch" in text:
        return "The Pitch"
    elif "Masters of Scale" in text:
        return "Masters of Scale"
    elif "Week in Startup" in text:
        return "This Week in Startups"
    elif "UI Breakfast" in text:
        return "UI Breakfast"
    elif "a16z" in text:
        # return "a16z Podcast"  # COMMENTED OUT: Use proper name extraction instead
        return "a16z Podcast"
    elif "Indie Hackers" in text:
        return "Indie Hackers"
    elif "Lenny" in text:
        return "Lenny's Podcast"
    
    # Default case: capitalize words
    return ' '.join(word.capitalize() for word in text.split())

def get_host_info(podcast_name):
    """Add host information for known podcasts"""
    hosts = {
        "TechCrunch Equity": "Alex Wilhelm, Natasha Mascarenhas",
        "How I Built This with Guy Raz": "Guy Raz",
        "Lex Fridman Podcast": "Lex Fridman",
        "My First Million": "Sam Parr, Shaan Puri",
        "The Twenty Minute VC": "Harry Stebbings",
        "The Pitch": "Josh Muccio",
        "Masters of Scale": "Reid Hoffman",
        "This Week in Startups": "Jason Calacanis",
        "UI Breakfast": "Jane Portman",
        # "a16z Podcast": "a16z Team",  # COMMENTED OUT: Use YAML config instead
        "Indie Hackers": "Courtland Allen",
        "Lenny's Podcast": "Lenny Rachitsky"
    }
    return hosts.get(podcast_name, "")

def extract_episode_title(filename, podcast_name):
    """Extract a cleaner episode title"""
    # Remove the podcast name part if present
    clean_filename = filename.replace(podcast_name.lower().replace(" ", "_"), "")
    
    # Remove common prefixes
    clean_filename = re.sub(r'^[_\-\s]+', '', clean_filename)
    
    # Generate a nicer title
    if "lex_ai_janna_levin" in filename:
        return "Janna Levin: AI, Physics, and the Future of Mind"
    elif "Equity_TCML" in filename:
        return "Rippling vs Deel and Corporate Espionage"
    elif "First_Million_HS" in filename:
        return "How to Retire with Millions and Pay $0 Taxes"
    elif "The_Pitch_VMP" in filename:
        return "CurieDx: AI Pocket Doctor"
    elif "VC_The_Trio" in filename:
        return "Tiger Global, OpenAI & Fundraising Strategies"
    elif "24d48b2f-f0c9" in filename:
        return "Chris Best and Hamish McKenzie on Building Substack"
    elif "ac6e1519-dc03" in filename:
        return "RJ Scaringe: Building Rivian Electric Vehicles"
    # Additional special cases can be added here
    
    # Default: clean up and format nicely
    words = clean_filename.replace('_', ' ').split()
    if words:
        return ' '.join(words).strip()
    else:
        return "Latest Episode"

# Process all transcript files
for jf in glob.glob("data/transcripts/*.json"):
    data = json.load(open(jf))
    
    # Extract filename without extension and path
    fname = os.path.basename(jf).replace(".json", "")
    
    # Get podcast name
    podcast_name = clean_name(fname.split('_')[0])
    
    # Extract episode title
    episode_title = extract_episode_title(fname, podcast_name)
    
    # Create rich metadata
    meta = {
        "podcast": podcast_name,
        "episode": episode_title,
        "date": "",
        "author": get_host_info(podcast_name)
    }
    
    # Update metadata
    data["meta"] = meta
    
    # Save updated file
    json.dump(data, open(jf, "w"))
    print(f"Enhanced metadata for {podcast_name} - {episode_title}")

print("Done enhancing transcript files")
