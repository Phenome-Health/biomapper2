"""Static checks on the nginx rate-limit config in deploy/nginx/.

These tests read the config files as text. They do not run nginx. They guard the
policy in deploy/README.md ("Rate limiting (nginx)"): both mapping mounts carry a
per-IP limit_req and a global limit_conn, rejections use the configured limit_*_status, biomapper-ui's
same-host address is exempt from the per-IP limit, and each limited location keeps
the full proxy setup (a nested location does not inherit it from `location /`).
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
ZONES = DEPLOY / "nginx" / "biomapper2-ratelimit-zones.conf"
LOCATIONS = DEPLOY / "nginx" / "biomapper2-ratelimit-locations.conf"
UNIT_FILE = DEPLOY / "biomapper2-api.service"

# Both mounts of mapping.router in src/biomapper2/api/main.py.
MAPPING_MOUNTS = ("/api/v1/map/", "/map/")
HOST_PUBLIC_IP = "35.161.242.62"  # Lightsail static IP, deploy/README.md
UPSTREAM = "http://127.0.0.1:8001"
WORKERS = 2  # --workers in deploy/biomapper2-api.service


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _blocks(text: str, keyword: str) -> dict[str, str]:
    """Map each `<keyword> <arg...> { body }` block to its body. Assumes no nested braces."""
    pattern = rf"\b{keyword}\s+([^{{;]+?)\s*\{{([^{{}}]*)\}}"
    return {m.group(1).strip(): m.group(2) for m in re.finditer(pattern, text)}


def _directives(body: str, name: str) -> list[str]:
    """Arguments of every `name ...;` directive in a block body."""
    return [m.group(1).strip() for m in re.finditer(rf"(?m)^\s*{name}\s+([^;]*);", body)]


@pytest.fixture(scope="module")
def zones() -> str:
    return _strip_comments(ZONES.read_text())


@pytest.fixture(scope="module")
def locations() -> dict[str, str]:
    return _blocks(_strip_comments(LOCATIONS.read_text()), "location")


def _req_zones(zones: str) -> dict[str, str]:
    """limit_req_zone name -> key."""
    out = {}
    for args in _directives(zones, "limit_req_zone"):
        key, *rest = args.split()
        name = next(a.split("=", 1)[1].split(":")[0] for a in rest if a.startswith("zone="))
        out[name] = key
    return out


def _conn_zones(zones: str) -> dict[str, str]:
    """limit_conn_zone name -> key."""
    out = {}
    for args in _directives(zones, "limit_conn_zone"):
        key, zone_arg = args.split()[:2]
        out[zone_arg.split("=", 1)[1].split(":")[0]] = key
    return out


def test_every_mapping_mount_has_a_location(locations):
    assert set(MAPPING_MOUNTS) <= set(locations), f"missing mounts: {set(MAPPING_MOUNTS) - set(locations)}"


def test_only_mapping_mounts_are_limited(locations):
    """Discovery and health stay on the existing `location /`, unthrottled."""
    assert set(locations) == set(MAPPING_MOUNTS)


@pytest.mark.parametrize("mount", MAPPING_MOUNTS)
def test_mount_has_per_ip_limit_req_on_declared_zone(mount, locations, zones):
    req_zones = _req_zones(zones)
    limit_req = _directives(locations[mount], "limit_req")
    assert limit_req, f"{mount} has no limit_req"
    zone_match = re.search(r"zone=(\S+)", limit_req[0])
    assert zone_match, f"{mount} limit_req names no zone: {limit_req[0]}"
    zone_name = zone_match.group(1)
    assert zone_name in req_zones, f"{mount} uses undeclared limit_req zone {zone_name}"
    # Per-IP via the exemption map, not keyed on a constant.
    assert req_zones[zone_name] == "$biomapper2_ratelimit_key"


# Pinned policy values (deploy/README.md "Rate limiting (nginx)"). Changing them is a
# policy change and should fail here first.
PER_IP_RATE = "1r/s"
BURST = "10"


def test_per_ip_zone_rate_is_pinned(zones):
    args = _directives(zones, "limit_req_zone")
    assert len(args) == 1
    assert re.search(rf"\brate={re.escape(PER_IP_RATE)}(\s|$)", args[0]), args[0]


@pytest.mark.parametrize("mount", MAPPING_MOUNTS)
def test_mount_limit_req_burst_and_nodelay_are_pinned(mount, locations):
    (limit_req,) = _directives(locations[mount], "limit_req")
    opts = limit_req.split()[1:]
    assert f"burst={BURST}" in opts, limit_req
    assert "nodelay" in opts, limit_req


@pytest.mark.parametrize("mount", MAPPING_MOUNTS)
def test_mount_holds_slot_until_upstream_finishes(mount, locations):
    """Client disconnect must not free the limit_conn slot while the engine still maps."""
    assert _directives(locations[mount], "proxy_ignore_client_abort") == ["on"]


@pytest.mark.parametrize("mount", MAPPING_MOUNTS)
def test_mount_has_global_limit_conn(mount, locations, zones):
    conn_zones = _conn_zones(zones)
    limit_conn = _directives(locations[mount], "limit_conn")
    assert limit_conn, f"{mount} has no limit_conn"
    zone_name, cap = limit_conn[0].split()
    assert conn_zones.get(zone_name) == "$server_name", "limit_conn zone must be global ($server_name)"
    assert int(cap) == WORKERS


@pytest.mark.parametrize("mount", MAPPING_MOUNTS)
def test_mount_rejects_with_429(mount, locations):
    body = locations[mount]
    assert _directives(body, "limit_req_status") == ["429"]
    assert _directives(body, "limit_conn_status") == ["429"]


def test_geo_exempts_host_ip_and_loopback(zones):
    geos = _blocks(zones, "geo")
    assert "$biomapper2_ratelimit_exempt" in geos
    lines = geos["$biomapper2_ratelimit_exempt"].replace(";", "\n").splitlines()
    entries = dict(line.split()[:2] for line in lines if line.split())
    assert entries.get("default") == "0"
    for addr in (f"{HOST_PUBLIC_IP}/32", "127.0.0.1/32"):
        assert entries.get(addr) == "1", f"{addr} is not exempt"


def test_exempt_hosts_get_empty_per_ip_key(zones):
    maps = _blocks(zones, "map")
    body = maps["$biomapper2_ratelimit_exempt $biomapper2_ratelimit_key"]
    assert re.search(r'(?m)^\s*1\s+""\s*;', body), "exempt hosts must map to an empty key"
    assert re.search(r"(?m)^\s*default\s+\$binary_remote_addr\s*;", body)


@pytest.mark.parametrize("mount", MAPPING_MOUNTS)
def test_mount_repeats_full_proxy_setup(mount, locations):
    body = locations[mount]
    assert _directives(body, "proxy_pass") == [UPSTREAM]
    assert _directives(body, "proxy_read_timeout") == ["300s"]
    headers = {h.split()[0]: h.split()[1] for h in _directives(body, "proxy_set_header")}
    assert headers == {
        "Host": "$host",
        "X-Real-IP": "$remote_addr",
        "X-Forwarded-For": "$proxy_add_x_forwarded_for",
        "X-Forwarded-Proto": "$scheme",
    }


def test_unit_file_workers_match_global_cap():
    assert f"--workers {WORKERS}" in UNIT_FILE.read_text()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Follow-up PR: bind uvicorn to 127.0.0.1 so nginx cannot be bypassed. Kept separate "
        "because deploy-api.yml deploys the unit file on merge to main. When that PR lands, "
        "this XPASSes and fails (strict): remove the xfail marker then."
    ),
)
def test_unit_file_binds_loopback_only():
    text = UNIT_FILE.read_text()
    assert "--host 127.0.0.1" in text
    assert "0.0.0.0" not in text
