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

   Confirm which KRAKEN build a deployment is serving with `curl $KESTREL_API_URL/health`
   (keyless), which reports `kg_version`, `kraken_package_version`, `biolink_version`, the build
   `git_commit` and the ingested `sources`; `/metagraph` alone does not identify the build.

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
   Add to `/etc/nginx/sites-available/biomapper`:
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

## Rate limiting (nginx)

The mapping routes (`/api/v1/map/*` and the `/map/*` alias, which reach the same router)
are rate-limited in nginx, not in the app. The handlers block the event loop, and an in-app
counter would be split across the two uvicorn workers. Discovery and health are not limited.

| File | Context | Installed to |
|------|---------|--------------|
| `deploy/nginx/biomapper2-ratelimit-zones.conf` | `http` | `/etc/nginx/conf.d/` |
| `deploy/nginx/biomapper2-ratelimit-locations.conf` | `server` (certbot block for `biomapper.expertintheloop.io`) | `/etc/nginx/snippets/` |

Policy:

- **Per IP:** `limit_req` at 1 r/s with `burst=10 nodelay`. Over the limit returns `429`.
- **Global:** `limit_conn` of 2 concurrent mapping requests for the whole server block (matches
  `--workers 2`). A third concurrent request gets an immediate `429`. It does not queue.
- **Exempt from the per-IP limit:** `127.0.0.1`, `::1`, and the host's own public IP
  `35.161.242.62`. biomapper-ui runs on this host and calls the public URL, so all UI users
  arrive from one address. Exempt hosts still count against the global cap of 2.

These are conservative starting values. Tune `rate`/`burst` through a PR to these files, not by
hand-editing the host.

### Apply (on the Lightsail host)

```bash
ssh -i ~/.ssh/lightsail-expert.pem ubuntu@35.161.242.62
cd ~/biomapper2 && git fetch origin main && git reset --hard origin/main   # the merged commit

# 0. Back up the current site config.
sudo cp /etc/nginx/sites-available/biomapper /etc/nginx/sites-available/biomapper.bak.$(date +%F)

# 1. Install the zones (http context) and the locations snippet.
sudo cp deploy/nginx/biomapper2-ratelimit-zones.conf /etc/nginx/conf.d/
sudo mkdir -p /etc/nginx/snippets
sudo cp deploy/nginx/biomapper2-ratelimit-locations.conf /etc/nginx/snippets/

# 2. Find the certbot-managed server block (listen 443 ssl; server_name biomapper.expertintheloop.io).
grep -n "server_name biomapper.expertintheloop.io" /etc/nginx/sites-enabled/*
sudoedit /etc/nginx/sites-available/biomapper
```

Inside that `server { ... }` block, add this line just before the existing `location / {`:

```nginx
    include /etc/nginx/snippets/biomapper2-ratelimit-locations.conf;
```

Check the existing `location / {` for directives beyond `proxy_pass` and the four
`proxy_set_header` lines (for example `client_max_body_size`, `proxy_buffering`,
`proxy_send_timeout`). Directives set inside `location /` are not inherited by the new
locations, so copy any such lines into both locations in the snippet. Directives set at
`server` level are inherited and need nothing.

```bash
# 3. Validate, then reload (graceful: in-flight requests finish).
sudo nginx -t && sudo systemctl reload nginx
```

Copy the snippet into `/etc/nginx/snippets/` rather than including it from `~/biomapper2`.
`deploy-api.yml` runs `git reset --hard origin/main` on the host, which would otherwise change
the live nginx config at the next reload without an `nginx -t`.

### Verify

Run these from a machine that is **not** the Lightsail host, since the host IP is exempt from
the per-IP limit. The burst omits the API key on purpose: nginx applies the limit before the
app sees the request, so the allowed requests return a fast `401` and never run a mapping.

```bash
URL=https://biomapper.expertintheloop.io

# 25 back-to-back requests from one IP: expect about 11 x 401 (1 + burst 10), then 429s.
for i in $(seq 1 25); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST "$URL/api/v1/map/entity" \
    -H "Content-Type: application/json" -d '{"name":"glucose","entity_type":"metabolite"}'
done | sort | uniq -c

# The /map alias shares the same per-IP bucket: immediately after the burst, expect 429.
curl -s -o /dev/null -w "%{http_code}\n" -X POST "$URL/map/entity" \
  -H "Content-Type: application/json" -d '{"name":"glucose","entity_type":"metabolite"}'

# Health is not limited and stays fast right after the burst: expect 200 well under a second.
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" "$URL/api/v1/health"
```

biomapper-ui on this host calls the engine at `http://localhost:8001` directly (its
`BIOMAPPER_BASE_URL`), so its traffic never passes through nginx and is not rate limited. The
`geo` exemption only matters if it is ever pointed at the public URL instead.

Reading 429s in the logs:

```bash
# This site logs with the `bm2` format (conf.d/biomapper2-keymap.conf) to its own file, which
# records status and consumer. Count and inspect rejections there:
grep -c 'status=429' /var/log/nginx/biomapper2.access.log
grep 'status=429' /var/log/nginx/biomapper2.access.log | tail
```

The rejections are logged at `warn`, but the host's `error_log` uses nginx's default `error`
level, so they do **not** appear in `/var/log/nginx/error.log`.

### Port 8001 must not be publicly reachable

The limit only holds if clients cannot reach uvicorn directly. The service binds
`127.0.0.1:8001` (`deploy/biomapper2-api.service`). Keep the Lightsail firewall (instance >
Networking > IPv4 and IPv6 firewalls) closed on 8001 as a second layer, and check from outside:

```bash
# From a machine that is NOT the Lightsail host: expect a timeout or refusal, not a response.
curl -m 5 http://35.161.242.62:8001/api/v1/health
```

Every local caller (nginx, biomapper-ui, the `deploy-api.yml` health check) uses `localhost`,
which resolves to `127.0.0.1` only on this host (`/etc/hosts` maps `::1` to `ip6-localhost`).
If that ever changes, a loopback-v4 bind would refuse `::1` callers.

`.github/workflows/deploy-api.yml` copies the unit file to the host and restarts the service on
every merge to `main` that touches `deploy/**`, so unit-file changes go live on merge. The nginx
steps above stay manual.

### Roll back

```bash
# 1. Remove the include line from the server block (or restore the backup from step 0):
#    sudo cp /etc/nginx/sites-available/biomapper.bak.<date> /etc/nginx/sites-available/biomapper
# 2. Then remove the installed files and reload.
sudo rm /etc/nginx/snippets/biomapper2-ratelimit-locations.conf
sudo rm /etc/nginx/conf.d/biomapper2-ratelimit-zones.conf
sudo nginx -t && sudo systemctl reload nginx
```

Remove the include line first. A `limit_req zone=...` that names a missing zone fails
`nginx -t`.

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
