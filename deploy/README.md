# Biomapper2 API Deployment

This directory contains deployment configuration for the Biomapper2 REST API.

## AWS Lightsail Deployment

The API is deployed to the same Lightsail instance as kraken-backend:

| Property | Value |
|----------|-------|
| Instance | `expert-in-the-loop-upgraded` |
| IP | `35.161.242.62` |
| SSH | `ssh -i ~/.ssh/lightsail-expert.pem ubuntu@35.161.242.62` |
| Port | `8001` |
| URL | `https://biomapper.expertintheloop.io` |

## Initial Setup

1. **Clone the repository on the server:**
   ```bash
   ssh -i ~/.ssh/lightsail-expert.pem ubuntu@35.161.242.62
   cd ~
   git clone https://github.com/Phenome-Health/biomapper2.git
   cd biomapper2
   ```

2. **Create environment file:**
   ```bash
   cat > .env << 'EOF'
   KESTREL_API_KEY=your-kestrel-api-key
   BIOMAPPER_API_KEY=your-biomapper-api-key
   EOF
   ```

3. **Install uv and dependencies:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   uv sync
   ```

4. **Install and start the service:**
   ```bash
   sudo cp deploy/biomapper2-api.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable biomapper2-api
   sudo systemctl start biomapper2-api
   ```

5. **Configure nginx:**
   Add to `/etc/nginx/sites-available/default`:
   ```nginx
   server {
       listen 80;
       server_name biomapper.expertintheloop.io;

       location / {
           proxy_pass http://127.0.0.1:8001;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

6. **Get SSL certificate:**
   ```bash
   sudo certbot --nginx -d biomapper.expertintheloop.io
   ```

## GitHub Actions Secrets

The deployment workflow requires these secrets:

- `LIGHTSAIL_HOST`: Server IP address
- `LIGHTSAIL_SSH_KEY`: Private SSH key for the server

## Service Management

```bash
# Check status
sudo systemctl status biomapper2-api

# View logs
sudo journalctl -u biomapper2-api -f

# Restart
sudo systemctl restart biomapper2-api

# Stop
sudo systemctl stop biomapper2-api
```

## Local Development

```bash
# Run locally
uv run uvicorn biomapper2.api.main:app --reload --port 8001

# Test
curl http://localhost:8001/api/v1/health
```
