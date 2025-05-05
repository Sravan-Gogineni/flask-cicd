🚀 Flask App Deployment with GitHub Actions and EC2
This project uses GitHub Actions to automatically deploy a Flask application to an EC2 instance every time changes are pushed to the main branch.

🛠️ Prerequisites
An AWS EC2 instance (Ubuntu preferred)

Flask app hosted in a GitHub repository

Flask app configured with systemd (e.g., flask-app.service)

Python virtual environment set up on EC2

SSH access from GitHub to your EC2 instance

Git installed and project cloned to EC2

🔐 GitHub Secrets
In your GitHub repository, go to Settings > Secrets and variables > Actions > New repository secret and add the following:

Secret Name	Description
EC2_SSH_KEY	The private key (PEM content) for the EC2 instance

📂 Project Structure on EC2
Make sure your EC2 instance has a structure like:

bash
Copy
Edit
~/flask-cicd/
├── app.py
├── requirements.txt
├── venv/
├── flask-app.service   # in /etc/systemd/system/
🧪 GitHub Actions Workflow
This workflow (.github/workflows/deploy.yml) runs on every push to the main branch.

yaml
Copy
Edit
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
        ssh -o StrictHostKeyChecking=no ubuntu@3.148.122.62 << 'EOF'
          cd ~/flask-cicd || { echo "Failed to change directory"; exit 1; }
          rm -rf __pycache__/app.cpython-310.pyc
          git pull origin main || { echo "Failed to pull from Git"; exit 1; }
          source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
          pip install -r requirements.txt || { echo "Failed to install requirements"; exit 1; }
          sudo systemctl daemon-reload || { echo "Failed to reload systemd"; exit 1; }
          sudo systemctl restart flask-app || { echo "Failed to restart Flask app"; exit 1; }
          sudo systemctl status flask-app || { echo "Flask app is not running"; exit 1; }
        EOF
🧾 Systemd Service (flask-app.service)
Create a systemd unit file at /etc/systemd/system/flask-app.service on your EC2 instance:

ini
Copy
Edit
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
Then run:

bash
Copy
Edit
sudo systemctl daemon-reload
sudo systemctl enable flask-app
sudo systemctl start flask-app
✅ Deployment Success
If everything is set up properly, pushing to main will:

Connect to your EC2 via SSH

Pull the latest code

Activate your virtual environment

Install/update dependencies

Restart your Flask app with systemctl
