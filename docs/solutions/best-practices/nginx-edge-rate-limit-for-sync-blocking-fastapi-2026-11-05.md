---
title: "Rate-limit a sync-blocking FastAPI engine at the nginx edge: per-IP limit_req + global limit_conn on every mount, held through client aborts"
date: 2026-11-05
category: best-practices
module: biomapper2
problem_type: best_practice
component: tooling
severity: high
related_components:
  - development_workflow
applies_when:
  - "async def routes call synchronous code, so the event loop blocks and an in-app limiter cannot answer and per-worker counters split"
  - "one router is mounted at more than one path prefix (e.g. /api/v1/map/ and a /map/ alias) and every mount needs the same limit"
  - "a global concurrency cap must track backend work, not client connections (limit_conn frees on client disconnect unless proxy_ignore_client_abort on)"
  - "a deploy workflow resets the repo checkout on the host, so nginx must not include config files directly from it"
  - "tightening a service bind (0.0.0.0 to 127.0.0.1) that the deploy workflow auto-applies and that local callers reach via localhost"
symptoms:
  - "prod API wedges under concurrent mapping load; one slow request occupies a whole uvicorn worker"
  - "clients could disconnect and resubmit past the global limit_conn cap while the engine kept working"
  - "429 rejections missing from error.log (logged at warn; the host error_log level is error)"
root_cause: config_error
resolution_type: config_change
tags:
  - nginx
  - rate-limiting
  - limit-conn
  - proxy-ignore-client-abort
  - fastapi-blocking-event-loop
  - uvicorn-bind-localhost
  - deploy-safety
  - lightsail
---

# Rate-limit a sync-blocking FastAPI engine at the nginx edge

## Context

Prod is one AWS Lightsail host. nginx 1.18 sits in front of uvicorn running `--workers 2` on
`:8001`. The mapping handlers are `async def` but call the synchronous mapper
(`mapper.map_entity_to_kg(...)` in `src/biomapper2/api/routes/mapping.py`), so each request blocks
its worker's event loop for the whole mapping. Under concurrent load prod wedges. Client-side
care (small batches, a health gate) was the only protection
(`reference_biomapper_prod_api_throughput`, auto memory [claude]).

Listing a public MCP server (the `biomapper` client, 2026-11-04) would add many independent
clients, so prod needed an **aggregate** limit first. An in-app limiter was ruled out:

- **The blocked loop can't answer a 429.** A worker that is busy mapping can't run limiter
  middleware.
- **Counters split across workers.** Each worker counts separately, so a cap of 2 really allows 4.

The router is also mounted twice in `src/biomapper2/api/main.py`, and both paths reach the same
handler:

```python
app.include_router(mapping.router, prefix="/api/v1/map", tags=["Mapping"])
app.include_router(mapping.router, prefix="/map", include_in_schema=False)
```

Shipped as fork #96 → org #130 → #131 (config) and fork #97 → org #133 → #134 (loopback bind and
README corrections). Live on prod since 2026-11-04.

## Guidance

1. **Limit at the proxy, not in the app,** whenever handlers block the loop or several worker
   processes serve one port.
2. **Use two limits.**
   - A per-IP `limit_req` gives fairness between clients.
   - A **global** `limit_conn` keyed on `$server_name` bounds total concurrency however many
     clients there are. Set its cap to the worker count.
3. **Limit every mount of the router.** A looser zone on an alias is a bypass. Each limited
   `location` must repeat the full proxy setup, because a more specific location does not inherit
   `proxy_*` from `location /`. That includes `proxy_read_timeout`, since the nginx default of 60s
   is shorter than a slow mapping.
4. **Set `proxy_ignore_client_abort on`.** By default nginx finalizes the request, and frees its
   `limit_conn` slot, as soon as the client disconnects, while the synchronous work keeps running
   upstream. A client could disconnect and resubmit past the cap. (Greptile P1, fixed in
   `1d43030`.) The cost is that an abandoned request holds its slot until the engine finishes, up
   to `proxy_read_timeout`.
5. **Exempt same-host callers from the per-IP limit only.** Use `geo` plus a `map` that yields an
   empty key, which `limit_req` does not count. Exempt callers still count against the global cap.
6. **Install copies of the config, not repo paths.** The deploy workflow runs
   `git reset --hard origin/main` on the host. An `include` pointing into the checkout would change
   live nginx at the next reload without anyone running `nginx -t`. Copy the zones to `conf.d/` and
   the locations to `snippets/`, and add only one `include` line to the certbot-managed server
   block.
7. **Close the direct-port bypass with a loopback bind, as its own PR.** `deploy-api.yml` copies
   the unit file and restarts the service on merge, so a bind change goes live on merge. Before
   that PR:
   - Confirm every local caller uses `localhost`.
   - Confirm `localhost` resolves to `127.0.0.1` only (`getent ahosts localhost`, `/etc/hosts`).
     If it resolves to `::1`, a v4-only bind refuses those callers, including the deploy's own
     health check.
8. **Know what the limit does not cover.** biomapper-ui on the same host calls
   `http://localhost:8001` directly. It never passes through nginx, so neither limit applies to it.
   The limits bound external traffic (MCP users, scripts, the public API), not total engine load.
9. **Pin the policy in static tests.** `tests/test_deploy_config.py` reads the config as text and
   asserts that:
   - every mount is limited;
   - rate, burst and `nodelay` match the policy;
   - the global cap equals the worker count;
   - rejections return 429;
   - `proxy_ignore_client_abort on` is set;
   - the geo exemption is present;
   - the full proxy setup is repeated in each location;
   - the unit file binds to loopback.

## Why This Matters

- **Without a global cap, a handful of clients wedge everything,** health checks included.
  Per-IP limits alone do not bound aggregate load.
- **An in-app limiter fails open** exactly when the loop is blocked, which is when you need it.
- **Each of these silently voids the limit, and none raises an error:**
  - a missed alias mount;
  - a slot freed on client abort;
  - a `0.0.0.0` bind;
  - checking a log that never records rejections.

## When to Apply

- Any FastAPI or Starlette service whose `async def` handlers do blocking CPU or I/O work.
- Multi-worker uvicorn or gunicorn behind nginx on one host.
- Before opening an endpoint to a wider or automated audience (MCP servers, public APIs).
- Whenever a router is mounted under more than one prefix.

## Examples

**Zones** (http context, `deploy/nginx/biomapper2-ratelimit-zones.conf` → `/etc/nginx/conf.d/`)

```nginx
geo $biomapper2_ratelimit_exempt {
    default           0;
    127.0.0.1/32      1;
    ::1/128           1;
    35.161.242.62/32  1;
}
map $biomapper2_ratelimit_exempt $biomapper2_ratelimit_key {
    1        "";
    default  $binary_remote_addr;
}
limit_req_zone $biomapper2_ratelimit_key zone=biomapper2_map_per_ip:10m rate=1r/s;
limit_conn_zone $server_name zone=biomapper2_map_global:1m;
```

**One limited location** (`deploy/nginx/biomapper2-ratelimit-locations.conf` →
`/etc/nginx/snippets/`; `/map/` is identical)

```nginx
location /api/v1/map/ {
    limit_req  zone=biomapper2_map_per_ip burst=10 nodelay;
    limit_conn biomapper2_map_global 2;
    limit_req_status  429;
    limit_conn_status 429;
    limit_req_log_level  warn;
    limit_conn_log_level warn;

    proxy_pass http://127.0.0.1:8001;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 300s;
    proxy_ignore_client_abort on;
}
```

**The two later fixes**

```diff
# 1d43030 — both locations
     proxy_read_timeout 300s;
+    proxy_ignore_client_abort on;

# 3bd3c74 — deploy/biomapper2-api.service (separate PR; auto-deploys on merge)
-... uvicorn biomapper2.api.main:app --host 0.0.0.0 --port 8001 --workers 2
+... uvicorn biomapper2.api.main:app --host 127.0.0.1 --port 8001 --workers 2
```

**Apply on the host, with rollback** (what was run 2026-11-04; the manual steps are in
`deploy/README.md`)

1. Record a baseline status for every site on the box. There were 11; analytics answers 401
   because of basic auth.
2. Back up the site file: `cp /etc/nginx/sites-available/biomapper{,.bak.<ts>-ratelimit}`. The
   file is `biomapper`, not `default` as the first README draft assumed.
3. Copy the zones file into `conf.d/` and the locations file into `snippets/`.
4. Insert `include /etc/nginx/snippets/biomapper2-ratelimit-locations.conf;` before the first
   `location / {` in the 443 block.
5. Run `nginx -t`.
   - If it fails: restore the backup, remove both copied files, and stop.
   - If it passes: `systemctl reload nginx`.
6. Re-check every site against the baseline.

To roll back later: remove the include line first, then the files, then run `nginx -t` and
reload.

**Verify from a machine that is not the host** (the host itself is exempt)

```bash
B=https://biomapper.expertintheloop.io
# Body '{}' fails validation in milliseconds (422), so the burst puts zero load on the engine.
for i in $(seq 1 25); do curl -s -o /dev/null -w '%{http_code} ' -X POST \
  -H 'Content-Type: application/json' -d '{}' $B/api/v1/map/entity; done; echo
#   observed: 12 x 422 then 429s (1 + burst 10, plus a refill at 1 r/s while the loop runs)
curl -s -o /dev/null -w '%{http_code}\n' -X POST -H 'Content-Type: application/json' -d '{}' $B/map/entity
#   429: the alias shares the bucket
curl -s -o /dev/null -w '%{http_code} %{time_total}s\n' $B/api/v1/health   # fast 200
curl -s -m 5 -o /dev/null -w '%{http_code}\n' http://35.161.242.62:8001/api/v1/health  # 000
# on the host:
ss -tlnp | grep :8001                                        # 127.0.0.1:8001 only
grep -c 'status=429' /var/log/nginx/biomapper2.access.log    # rejections live here
```

**The rejections are not in `error.log`.** The limits log at `warn`, but the host's `error_log`
uses nginx's default `error` level, so those lines are dropped. This site logs to its own file,
`biomapper2.access.log`, in the `bm2` format (`conf.d/biomapper2-keymap.conf`), and that file
records `status=429`.

**CI gotchas on the PRs**

- **pyright on regex matches:** `reportOptionalMemberAccess` fires on `re.search(...).group(1)`.
  Guard the match first.
- **The prose-figure guard:** the repo's `test_no_measured_figures_in_prose` flags any number with
  three or more digits in a docstring or comment, and "429" trips it. Name the directive instead
  ("the configured `limit_*_status`"), per
  `best-practices/audit-instruments-backing-published-claims-2026-08-05.md`.

## Related

- `deploy/README.md`, section "Rate limiting (nginx)": the operator runbook.
- Commits `97265d8` (initial config), `1d43030` (abort fix and pinned policy), `3bd3c74` (loopback
  bind and README corrections), and `7307e01` (enable the named site; state what the limits cover).
- Vault `docs/solutions/workflow-issues/unversioned-deploy-config-drift-2026-07-06.md` and
  `docs/solutions/integration-issues/agent-sdk-model-upgrade-bundled-cli-version-gate-2026-10-27.md`:
  the same git-reset deploy pattern. Check the running process, not the checkout.
- kraken-chatbot `docs/solutions/best-practices/clerk-auth-react-fastapi-integration-2026-05-06.md`:
  never overwrite certbot-managed nginx configs in deploy scripts. That is why only an `include`
  line is added here.
- `biomapper` client `docs/solutions/best-practices/mcp-registry-pypi-launch-contract-2026-11-04.md`:
  the MCP listing this limit was a prerequisite for.
