"""
Configuration settings for biomapper2.

Customize these values to change API endpoints, model versions, and logging behavior.
"""

import contextlib
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load environmental variables (secrets)


# Set up our general cache directory (e.g., for requests cache, biolink)
PROJECT_ROOT = Path(__file__).parents[2]
CACHE_DIR = PROJECT_ROOT / "cache"
# 0700: the requests_cache databases under here record request headers, so treat the directory as
# secret-bearing. `mode` only applies when mkdir creates the directory, so chmod an existing one
# too (best-effort — a read-only mount or foreign owner must not break import).
CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
with contextlib.suppress(OSError):
    CACHE_DIR.chmod(0o700)

# Header and query-parameter names requests_cache must redact before writing a response to disk.
#
# requests_cache ships a default list that already contains "X-API-KEY", but it matches
# CASE-SENSITIVELY (`filter_sort_dict` sorts a plain dict, discarding the CaseInsensitiveDict
# semantics), and we send the header spelled "X-API-Key". That one-character mismatch persisted the
# Kestrel credential in cleartext into every cached record. Enumerate the casings we actually send
# rather than relying on the library default.
CACHE_IGNORED_PARAMETERS = [
    "X-API-Key",
    "X-API-KEY",
    "x-api-key",
    "Authorization",
    "api_key",
    "access_token",
]

# KG API configuration — override via KESTREL_API_URL in .env.
# Defaults to the PUBLIC Kestrel, which needs no API key and serves the KRAKEN build the
# published benchmarks pin. The internal endpoint (kestrel.nathanpricelab.com/api) is a
# different, older build — GET /metagraph on each reports the graph and version it serves.
#
# Named separately so request construction can refuse to send credentials here: an environment
# that set KESTREL_API_KEY for the internal endpoint and never set KESTREL_API_URL would
# otherwise start leaking that key to a third-party host the moment this default changed.
PUBLIC_KESTREL_API_URL = "https://kestrel.krakenkg.com/api"
KESTREL_API_URL = os.getenv("KESTREL_API_URL", PUBLIC_KESTREL_API_URL)


def get_kestrel_api_url() -> str:
    """Return the current Kestrel API URL, reading os.environ on every call.

    Unlike the module-level KESTREL_API_URL constant (captured at import time),
    this function reflects any os.environ overrides applied after import — e.g.
    the --kestrel-url pytest option used in KG regression testing.

    The fallback is PUBLIC_KESTREL_API_URL — the SAME default as the constant above — and that is
    load-bearing, not tidiness. This function used to carry its own hardcoded default. It was
    written when the repo default was the internal host, and when that default was promoted to
    public (#78) only the constant moved. The two then disagreed, and because the client resolves
    through this function, an unset KESTREL_API_URL still reached the INTERNAL host while
    ``config.KESTREL_API_URL`` reported public. Keep exactly one place that decides the default.
    """
    return os.environ.get("KESTREL_API_URL", PUBLIC_KESTREL_API_URL)


# Biolink model version
BIOLINK_VERSION_DEFAULT = "4.2.5"

# Level of logging messages to display (DEBUG, INFO, WARNING, ERROR, or CRITICAL)
LOG_LEVEL = "INFO"

# Secrets (from environment variables)


def get_kestrel_api_key() -> str | None:
    """The Kestrel API key, or None when unset.

    An unset key is not an error: the default (public) endpoint requires no authentication, so
    requests simply go out without the header. Endpoints that do require it answer 401, which
    ``kestrel_request`` surfaces with a pointer back to ``KESTREL_API_KEY``.

    Deliberately not memoized. ``os.getenv`` is a dict lookup, and the previous cache keyed on
    ``is None`` — which, once an unset key started returning None, only ever took effect when a key
    WAS present. That froze the value for the process lifetime, so a test or caller changing the
    environment mid-run got a result that depended on call order.
    """
    return os.getenv("KESTREL_API_KEY") or None


# Default per-request timeout, in seconds, for every Kestrel call. The kwarg always passed through
# to the transport; what was missing was a DEFAULT, so the mapping-path callers that supply none had
# no timeout at all and a wedged request could hang a run indefinitely. Sized from the successful-
# request duration distribution recorded in
# studies/analysis/results/request_timeout_derivation.json (field: recommended_timeout_s)
# and asserted against it in the test suite -- a default UNDER the server's own limit is worse than
# none, because it converts a recoverable server error into a client-side abort.
KESTREL_REQUEST_TIMEOUT_S = 180

# Bisect-on-5xx. Ships DORMANT. Its diagnosis -- that server errors are determined by payload
# CONTENT rather than by load or timing -- is not yet confirmed by the gated single-request
# diagnostic. If the cause turns out to be load, bisecting amplifies it: the failing chunk is
# resubmitted as a tree of sub-chunks against a shared service with no rate limiting on either side.
# Enable only after the diagnostic returns, and never as a default.
KESTREL_BISECT_ON_5XX_ENABLED = False

# Bisect budgets, in REQUEST VOLUME rather than recursion depth. Depth is bounded near ten by
# construction and bounds nothing that matters; volume is what a shared, unrated service notices.
# Bisect composes with the retry ladder multiplicatively, and several independent bad items cost
# order m*log(N/m) nodes per chunk per vocabulary per dataset. Every cap fails LOUD.
KESTREL_BISECT_MAX_REQUESTS = 200  # per dataset
KESTREL_BISECT_MAX_WALL_CLOCK_S = 600.0
KESTREL_BISECT_MAX_CONSECUTIVE_FAILURES = 12
KESTREL_BISECT_MIN_INTER_REQUEST_DELAY_S = 1.0
# The retry ladder is disabled inside bisect: a node gets a single attempt. Sequential is not the
# same as polite, and the ladder's backoff sleeps turn one poison item into minutes of load.
KESTREL_BISECT_MAX_RETRIES = 0

# Batching for Kestrel API requests (to prevent timeouts on large datasets)
KESTREL_BATCHING_ENABLED = True  # Set to False to disable batching (for performance testing)
KESTREL_BATCH_SIZE_SEARCH = 1000  # For text-search, vector-search, hybrid-search
KESTREL_BATCH_SIZE_CANONICALIZE = 2000  # For canonicalize endpoint

# Structure (InChIKey) fallback services for the resolver's connectivity test. Used only on the
# small-molecule ChEBI conflict path when a node carries no KG InChIKey (see StructureResolver).
MW_INCHIKEY_URL = "https://www.metabolomicsworkbench.org/rest/refmet/name"  # /{name}/inchi_key
PUBCHEM_INCHIKEY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"  # /{name}/property/InChIKey/JSON
STRUCTURE_LOOKUP_TIMEOUT_S = 3  # per external structure call; mirrors the RefMet /match timeout

# Tier B of the resolution certificate: independent structure evidence for the QUERY NAME.
#
# OFF by default, and that default is part of the contract rather than a deployment convention.
# Tier A is zero-I/O and reads only what the graph already asserts about the committed node; turning
# Tier B on moves external calls from a small conflict subset to every unique query name in a run,
# against rate-limited services, and changes what the emitted certificate state means. An operator
# enables it deliberately for a supervised sweep. See core/tier_b.py.
TIER_B_ENABLED = os.getenv("BIOMAPPER2_TIER_B_ENABLED", "").strip().lower() in {"1", "true", "yes"}
TIER_B_MIN_INTERVAL_S = 0.25  # minimum spacing between outbound Tier B calls (PUG-REST is throttled)
TIER_B_MAX_ATTEMPTS = 3  # attempts per hop before recording lookup_failed
TIER_B_BACKOFF_BASE_S = 0.5  # first backoff; doubles per retry
# Floor on Tier B's own resolution rate below which the corroboration curve is refused rather than
# published: the endpoints are EXACT-name lookups while the annotator matches fuzzily, so a low rate
# means the verdicts were computed on a biased easy subset.
TIER_B_MIN_RESOLUTION_RATE = 0.5

# Structure-guided re-resolution of conflated KG commits (see core/resolver.py:reresolve_on_contradiction).
#
# OFF by default, and INERT unless TIER_B_ENABLED: re-resolution keys on a CONTRADICTED certificate,
# which only Tier B can produce, so enabling this without Tier B does nothing. The Mapper asserts the
# dependency and logs at build time. When on (and Tier B on), a contradiction triggers a single
# structure-guided attempt to swap the conflated node for the correct distinct candidate; a
# still-contradicted swap is a logged REFUSE, never a recursion. A gated production change: the
# falsifiable benchmark that measures precision/coverage lives in the separate benchmark axis.
RERESOLUTION_ENABLED = os.getenv("BIOMAPPER2_RERESOLUTION_ENABLED", "").strip().lower() in {"1", "true", "yes"}

# Human-preference re-ranking for gene/protein resolution (see docs/plans HGNC plan).
# When prefer_human is active, hybrid-search retrieves this many candidates (instead of 1) so the
# human node — which often ranks below the wrong-species ortholog — is actually returned. Live spike
# (2026-06-15) found recoverable human nodes at rank ~#4; 20 gives ample margin.
HYBRID_SEARCH_LIMIT = 20
# Human-only CURIE prefixes. HGNC assigns IDs only to human genes, so its presence in a hybrid-search
# row's `prefixes` marks the human node. Any prefix added here must itself be human-exclusive.
HUMAN_MARKER_PREFIXES = {"HGNC"}

# Kill switch for the curated gene-symbol fallback bridge (see core/gene_symbol_resolver.py). When True
# (default), gene/protein resolution misses for the curated drug-conflated symbols are resolved via the
# deterministic non-search fallback. Set False to disable the bridge without a code revert.
GENE_SYMBOL_FALLBACK_ENABLED = True

# Per-category preferred (canonical) namespace prefixes for the prefer_canonical re-ranking. Within a
# Biolink category, hybrid-search ranks across all namespaces at once, so a non-canonical same-text node
# (UMLS/ICD/KEGG/PANTHER) frequently outranks the canonical one. These prefixes mark the canonical node so
# the annotator can prefer it (see core/annotators/kestrel_hybrid.py:_select_canonical). Keys are Biolink
# categories; the engine expands each via get_descendants so subcategories inherit the policy. Gene/protein
# are intentionally absent — they use HUMAN_MARKER_PREFIXES / prefer_human instead.
#
# Prefix strings are the *actual* Kestrel KG prefixes, verified live 2026-06-18 against hybrid-search rows
# (e.g. RefMet is "RM", not "REFMET"). A wrong string is a silent no-op (the filter matches nothing).
CATEGORY_PREFERRED_NAMESPACES: dict[str, set[str]] = {
    "biolink:SmallMolecule": {"CHEBI", "HMDB", "RM"},
    "biolink:Disease": {"MONDO"},
}

# Per-category Biolink acceptance root for the category *validator* (see
# core/annotators/kestrel_hybrid.py:_is_on_category). Hybrid search ranks across categories as well as
# namespaces, so a metabolite query can commit a node that is not a molecule at all: a sizeable slice of
# metabolite-arm commits carry no chemical category, dominated by biolink:PhenotypicFeature (the EFO
# "…measurement" nodes) and by Protein/Gene typings.
#
# No figure is restated here on purpose. Every number behind this policy is emitted by
# `studies/analysis/off_category_audit.py` into its committed artifact under
# studies/analysis/results/; read the fields rather than a comment that can drift:
#   - metabolite_total / per_dataset  — the off-category rate, and how much of it each dataset carries
#   - off_category_composition_by_category — which Biolink types dominate the refused population
#   - protein_gene_refusal_cost — what the guard COSTS: whether any refused node was the right compound
#     wearing a wrong Biolink type (the peptide-metabolite worry: glutathione/carnosine/gamma-glutamyl-X)
#   - adjudicator_positive_control — proof that the cost measurement could have found a nonzero answer
#
# Semantics, and note the two sides are expanded *separately*:
#   - The KEY is expanded via get_descendants so subcategories of a configured job category inherit the
#     same policy (mirrors CATEGORY_PREFERRED_NAMESPACES).
#   - The VALUE is the acceptance ROOT and is expanded via get_descendants on its own. It is deliberately
#     one level up from the key: descendants('biolink:ChemicalEntity') is 12 categories, so a legitimately
#     broader typing (ChemicalEntity, MolecularMixture, Drug) survives while Protein/Polypeptide/
#     PhenotypicFeature/Pathway/MolecularActivity do not.
#
# Gene/protein are intentionally absent, and the gene arm is the control that proves why: nearly every
# hgnc commit is "off-category" relative to a chemical root, exactly as it should be (artifact field
# per_dataset.hgnc). Pointing a chemical acceptance set at the gene path would refuse almost all of it,
# so that path stays unfiltered. Any category with no entry here is likewise unfiltered — including the
# biolink:NamedThing that standardize_entity_type falls back to for an unrecognized entity type.
CATEGORY_ACCEPTED_ROOTS: dict[str, str] = {
    "biolink:SmallMolecule": "biolink:ChemicalEntity",
}
