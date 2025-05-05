# Deploy Flask App to EC2 Using GitHub Actions

## Introduction

This guide explains how to automate the deployment of a Flask application to an AWS EC2 instance using GitHub Actions.

## Prerequisites

### EC2 Instance

- An EC2 Ubuntu instance set up with a Flask app
- Python virtual environment created
- Git installed and the project cloned on the instance

### SSH Access

- A PEM-format SSH private key to connect to your EC2 instance

## GitHub Repository Configuration

### GitHub Secrets

You must add the following secret to your repository:

- EC2_SSH_KEY — your EC2 instance's private key

To add a secret:

1. Go to your GitHub repository.
2. Click on "Settings".
3. Navigate to "Secrets and variables" > "Actions".
4. Click "New repository secret".
5. Add `EC2_SSH_KEY` and paste your PEM key content.

## EC2 Project Directory Structure

Your EC2 instance should have this directory structure:

```
/home/ubuntu/flask-cicd/
├── app.py
├── requirements.txt
├── venv/
```

## GitHub Actions Workflow File

### Workflow File Location

The workflow file is already located at:

```
.github/workflows/deploy.yml
```

### Workflow Content

```yaml
name: Deploy Flask to EC2

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Setup SSH
      uses: webfactory/ssh-agent@v0.7.0
      with:
        ssh-private-key: ${{ secrets.EC2_SSH_KEY }}

    - name: Deploy to EC2
      run: |
        ssh -o StrictHostKeyChecking=no ubuntu@<YOUR_EC2_PUBLIC_IP> << 'EOF'
          cd ~/flask-cicd || { echo "Failed to change directory"; exit 1; }
          rm -rf __pycache__/app.cpython-310.pyc
          git pull origin main || { echo "Failed to pull from Git"; exit 1; }
          source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
          pip install -r requirements.txt || { echo "Failed to install requirements"; exit 1; }
          sudo systemctl daemon-reload || { echo "Failed to reload systemd"; exit 1; }
          sudo systemctl restart flask-app || { echo "Failed to restart Flask app"; exit 1; }
          sudo systemctl status flask-app || { echo "Flask app is not running"; exit 1; }
        EOF
```

Replace `<YOUR_EC2_PUBLIC_IP>` with your actual EC2 instance IP.

## Setting Up systemd on EC2

### Create the Service File

Create a file at `/etc/systemd/system/flask-app.service` with the following content:

```ini
[Unit]
Description=Gunicorn instance to serve Flask app
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/flask-cicd
Environment="PATH=/home/ubuntu/flask-cicd/venv/bin"
ExecStart=/home/ubuntu/flask-cicd/venv/bin/gunicorn -w 3 app:app -b 0.0.0.0:5000

[Install]
WantedBy=multi-user.target
```

### Enable and Start the Service

Run the following commands on the EC2 instance:

```bash
sudo systemctl daemon-reload
sudo systemctl enable flask-app
sudo systemctl start flask-app
```

## How It Works

1. You push code to the `main` branch on GitHub.
2. GitHub Actions connects to your EC2 instance over SSH.
3. It pulls the latest code and activates your Python virtual environment.
4. It installs updated dependencies.
5. It restarts the Flask app using `systemd`.

## Monitoring and Debugging

### View Logs

Use the following command to check Flask app logs:

```bash
sudo journalctl -u flask-app -f
```

### Common Issues

- Ensure permissions are set correctly on your PEM file and EC2 directory.
- Confirm that `git`, `python`, and `systemd` are installed and configured correctly.
