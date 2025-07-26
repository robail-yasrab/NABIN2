# 🚀 GitHub Deployment Guide

This guide will help you deploy your Multi-Omic Metabolic Pathway Analysis project to GitHub.

## 📋 Prerequisites

### 1. Install Git
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **macOS**: `brew install git` or download from the website
- **Linux**: `sudo apt-get install git` (Ubuntu/Debian) or `sudo yum install git` (CentOS/RHEL)

### 2. Create GitHub Account
- Go to [github.com](https://github.com) and create an account if you don't have one

### 3. Create a New Repository on GitHub
1. Log into GitHub
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name your repository (e.g., `multi-omic-pathway-analysis`)
5. Add description: "Transformer-based deep learning for multi-omic metabolic pathway bottleneck analysis"
6. Choose visibility (Public or Private)
7. **Don't** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## 🔧 Method 1: Automated Setup (Recommended)

We've created an automated script to handle the entire process:

```bash
# Run the automated setup script
python setup_git.py
```

The script will:
- ✅ Check if Git is installed and configured
- ✅ Initialize Git repository
- ✅ Create .gitignore file
- ✅ Add and commit all files
- ✅ Set up remote origin
- ✅ Push to GitHub

### Follow the prompts:
1. **Repository URL**: Paste your GitHub repository URL (e.g., `https://github.com/yourusername/multi-omic-pathway-analysis.git`)
2. **Commit message**: Press Enter for default or type custom message
3. **Branch name**: Press Enter for 'main' or specify different branch

## 🔧 Method 2: Manual Setup

If you prefer to do it manually or the automated script fails:

### Step 1: Configure Git (First time only)
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"
```

### Step 2: Initialize Repository
```bash
# Navigate to your project directory
cd path/to/your/project

# Initialize Git repository
git init

# Add remote origin (replace with your repository URL)
git remote add origin https://github.com/yourusername/multi-omic-pathway-analysis.git
```

### Step 3: Add and Commit Files
```bash
# Add all files
git add .

# Commit files
git commit -m "Initial commit: Multi-omic pathway analysis project"
```

### Step 4: Push to GitHub
```bash
# Set main branch and push
git branch -M main
git push -u origin main
```

## 🔐 Authentication Options

### Option 1: Personal Access Token (Recommended)
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` permissions
3. Use token as password when prompted

### Option 2: SSH Keys
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key to GitHub
cat ~/.ssh/id_ed25519.pub
```

Then add the public key to your GitHub account in Settings → SSH and GPG keys.

## 🔄 Subsequent Updates

After initial setup, use the quick push feature:

```bash
# Quick push with default message
python setup_git.py push

# Quick push with custom message
python setup_git.py push "Added new feature for bottleneck analysis"
```

Or manually:
```bash
git add .
git commit -m "Your commit message"
git push
```

## 📂 Project Structure on GitHub

Your repository will contain:
```
multi-omic-pathway-analysis/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── setup_git.py                   # Git automation script
├── test_installation.py           # Installation verification
├── DEPLOYMENT_GUIDE.md           # This guide
├── src/                          # Source code
│   ├── data/                     # Data handling modules
│   ├── models/                   # Deep learning models
│   ├── training/                 # Training utilities
│   ├── analysis/                 # Pathway analysis tools
│   └── utils/                    # Utility functions
├── examples/                     # Example usage
├── configs/                      # Configuration files
└── .gitignore                    # Git ignore rules
```

## 🏷️ Adding Repository Details

After successful deployment, enhance your repository:

### 1. Edit Repository Description
- Go to your repository on GitHub
- Click the gear icon next to "About"
- Add description: "Transformer-based deep learning framework for integrating multi-omic data to identify metabolic pathway bottleneck genes"
- Add topics: `deep-learning`, `pytorch`, `bioinformatics`, `multi-omics`, `metabolic-pathways`, `transformer`, `bottleneck-analysis`

### 2. Create Issues (Optional)
- Click "Issues" tab
- Create issues for planned features or improvements

### 3. Add Documentation
Consider adding these additional files:
- `CONTRIBUTING.md` - Guidelines for contributors
- `LICENSE` - Open source license
- `CHANGELOG.md` - Version history

## 🛠️ Troubleshooting

### Common Issues:

**1. "Permission denied" error:**
- Check authentication (Personal Access Token or SSH key)
- Verify repository URL is correct
- Ensure you have write permissions to the repository

**2. "Repository not found" error:**
- Verify the repository exists on GitHub
- Check the repository URL spelling
- Ensure the repository is accessible with your account

**3. "fatal: not a git repository" error:**
- Make sure you're in the correct directory
- Run `git init` to initialize the repository

**4. Large file warnings:**
- Git has a 100MB file size limit
- Use Git LFS for large model files
- Add large files to `.gitignore` if not needed in repository

### Getting Help:
- Check the [GitHub Documentation](https://docs.github.com/)
- Visit [git-scm.com](https://git-scm.com/doc) for Git documentation
- Reference: [GeeksforGeeks Git Tutorial](https://www.geeksforgeeks.org/how-to-deploy-python-project-on-github/)

## 🎉 Success!

Once deployed, your project will be:
- ✅ Backed up on GitHub
- ✅ Accessible from anywhere
- ✅ Ready for collaboration
- ✅ Version controlled
- ✅ Discoverable by the research community

Share your repository URL with collaborators and the research community!

## 📞 Need Help?

If you encounter any issues:
1. Check this guide first
2. Run `python test_installation.py` to verify setup
3. Check GitHub's documentation
4. Create an issue in your repository for tracking

Happy coding! 🧬🔬⚡ 