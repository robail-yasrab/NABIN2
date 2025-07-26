#!/usr/bin/env python3
"""
Git Setup and Push Automation Script
Based on: https://github.com/EkalavyanS/GITPUSH and https://www.geeksforgeeks.org/how-to-deploy-python-project-on-github/

This script helps automate the process of setting up Git repository and pushing to GitHub.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr.strip()}")
        return False


def check_git_installed():
    """Check if Git is installed."""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✅ Git is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed. Please install Git first:")
        print("   - Windows: Download from https://git-scm.com/download/win")
        print("   - macOS: brew install git")
        print("   - Linux: sudo apt-get install git")
        return False


def check_git_config():
    """Check and setup Git configuration."""
    try:
        # Check if user name is configured
        subprocess.run(["git", "config", "user.name"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email"], check=True, capture_output=True)
        print("✅ Git is configured with user details")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Git user configuration not found")
        setup_git_config()
        return True


def setup_git_config():
    """Setup Git user configuration."""
    print("\n📝 Setting up Git configuration...")
    
    name = input("Enter your full name: ").strip()
    email = input("Enter your email address: ").strip()
    
    if name and email:
        run_command(f'git config --global user.name "{name}"', "Setting Git username")
        run_command(f'git config --global user.email "{email}"', "Setting Git email")
        print("✅ Git configuration completed")
    else:
        print("❌ Invalid name or email provided")
        return False
    return True


def create_gitignore():
    """Create a comprehensive .gitignore file for Python projects."""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Project specific
results/
checkpoints/
*.pth
*.pt
*.model
.DS_Store
*.tmp
*.temp

# Data files (uncomment if you don't want to track large data files)
# *.csv
# *.h5
# *.hdf5
# *.pkl
# *.pickle
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ Created .gitignore file")


def init_repository():
    """Initialize Git repository."""
    if os.path.exists('.git'):
        print("✅ Git repository already exists")
        return True
    
    success = run_command("git init", "Initializing Git repository")
    if success:
        print("✅ Git repository initialized")
    return success


def add_and_commit_files(commit_message="Initial commit: Multi-omic pathway analysis project"):
    """Add and commit all files."""
    print("\n📦 Adding and committing files...")
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists('.gitignore'):
        create_gitignore()
    
    # Add all files
    if not run_command("git add .", "Adding all files to staging"):
        return False
    
    # Commit files
    if not run_command(f'git commit -m "{commit_message}"', "Committing files"):
        return False
    
    print("✅ Files committed successfully")
    return True


def setup_remote_and_push(remote_url, branch="main"):
    """Setup remote origin and push to GitHub."""
    print(f"\n🚀 Setting up remote and pushing to GitHub...")
    
    # Add remote origin
    if not run_command(f"git remote add origin {remote_url}", "Adding remote origin"):
        # If remote already exists, try to set the URL
        if not run_command(f"git remote set-url origin {remote_url}", "Updating remote origin URL"):
            return False
    
    # Rename branch to main if needed
    run_command(f"git branch -M {branch}", f"Setting branch to {branch}")
    
    # Push to remote
    if not run_command(f"git push -u origin {branch}", f"Pushing to origin/{branch}"):
        print("\n⚠️  Push failed. This might be due to:")
        print("   1. Authentication issues (need personal access token)")
        print("   2. Repository doesn't exist on GitHub")
        print("   3. No write permissions to the repository")
        print("\nPlease check your GitHub repository and authentication settings.")
        return False
    
    print("✅ Successfully pushed to GitHub!")
    return True


def main():
    """Main function to orchestrate the Git setup and push process."""
    print("🧬 Multi-Omic Analysis - GitHub Setup Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_git_installed():
        return False
    
    if not check_git_config():
        return False
    
    # Get repository details from user
    print("\n📋 Repository Setup")
    print("Please provide your GitHub repository details:")
    
    remote_url = input("Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): ").strip()
    if not remote_url:
        print("❌ Repository URL is required")
        return False
    
    commit_message = input("Enter commit message (press Enter for default): ").strip()
    if not commit_message:
        commit_message = "Initial commit: Multi-omic pathway analysis project"
    
    branch = input("Enter branch name (press Enter for 'main'): ").strip()
    if not branch:
        branch = "main"
    
    # Initialize repository
    if not init_repository():
        return False
    
    # Add and commit files
    if not add_and_commit_files(commit_message):
        return False
    
    # Setup remote and push
    if not setup_remote_and_push(remote_url, branch):
        return False
    
    print("\n🎉 Repository successfully set up and pushed to GitHub!")
    print(f"🔗 Your repository: {remote_url}")
    
    # Show next steps
    print("\n📚 Next Steps:")
    print("1. Visit your GitHub repository to verify the upload")
    print("2. Add collaborators if needed")
    print("3. Set up GitHub Actions for CI/CD (optional)")
    print("4. Create issues and project boards for tracking")
    print("5. Add repository topics and description for discoverability")
    
    return True


def quick_push(commit_message="Update project files"):
    """Quick function for subsequent pushes after initial setup."""
    print(f"\n🔄 Quick Push: {commit_message}")
    
    if not run_command("git add .", "Adding changed files"):
        return False
    
    if not run_command(f'git commit -m "{commit_message}"', "Committing changes"):
        return False
    
    if not run_command("git push", "Pushing to remote"):
        return False
    
    print("✅ Changes pushed successfully!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Quick push mode
        if sys.argv[1] == "push":
            message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Update project files"
            quick_push(message)
        else:
            print("Usage:")
            print("  python setup_git.py          # Full setup")
            print("  python setup_git.py push     # Quick push")
            print("  python setup_git.py push 'commit message'  # Quick push with message")
    else:
        # Full setup mode
        main() 