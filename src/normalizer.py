import re
import pandas as pd
from rapidfuzz import fuzz

# Tokens that indicate trim level, package, body style, or drivetrain —
# not part of the base model identity.
TRIM_TOKENS = frozenset({
    # Trim / package designations
    'ex', 'exl', 'lx', 'se', 'sel', 'le', 'xle', 'xls', 'xlt',
    'gle', 'glx', 'gli', 'gti', 'gt', 'gts', 'rs', 'ss', 'ls', 'lt', 'ltz',
    'z71', 'z28', 'st', 'svt', 'sv', 'sl', 'sr', 'sr5', 'trd',
    'dx', 'gl', 'gls', 'si',
    'sle', 'slt', 'slt1', 'slt2', 'denali',            # GMC
    'sahara', 'rubicon', 'laredo', 'overland',           # Jeep
    'trailhawk', 'altitude', 'latitude', 'longitude',    # Jeep
    'sport-s', 'sport-t', 'high',                        # misc
    'platinum', 'titanium', 'signature', 'premier', 'premium', 'preferred',
    'limited', 'touring', 'base', 'luxury', 'classic', 'deluxe', 'custom',
    'sport', 'edition', 'special', 'ultra', 'pro', 'plus', 'max',
    'advance', 'advanced',
    # Hyphenated trim variants (treated as a single token)
    'ex-l', 'ex-t', 'sr-5', 'gt-s', 'trd-pro',
    # Drivetrain
    'awd', 'fwd', 'rwd', '4wd', '4x4', '2wd',
    # Engine / fuel
    'v4', 'v6', 'v8', 'i4', 'i6', 'turbo', 'diesel', 'hybrid', 'electric',
    # Body styles
    'sedan', 'sdn', 'coupe', 'hatchback', 'wagon', 'convertible', 'pickup',
    'utility', 'suv',
    # Cab / bed descriptors (trucks)
    'cab', 'crew', 'crewmax', 'extended', 'regular', 'double', 'quad', 'king',
    'access', 'supercab', 'supercrew', 'shortbed', 'longbed',
    'srw', 'drw',                     # single/dual rear wheel
    'super', 'duty',                  # "Super Duty" is a Ford trim line
    # Door counts
    '2d', '4d', '2dr', '4dr',
    # Dodge / Chrysler / SRT trims
    'sxt', 'rt', 'srt', 'srt8', 'scat', 'pack', 'mopar', 'rallye',
    # Misc
    'new', 'model', 'package',
})

_NUM_RE = re.compile(r'\d+')


def strip_trims(name: str) -> str:
    """
    Remove trim-level tokens from a model string, keeping the base model name.
    Does NOT split on dashes — 'cr-v' and 'hr-v' survive intact.
    Hyphenated trim combos like 'ex-l' are in TRIM_TOKENS and stripped as a unit.
    """
    name = str(name).lower().strip()
    if name in ('nan', ''):
        return ''

    base = []
    for token in name.split():
        clean = re.sub(r'[^\w\-]', '', token)   # keep alphanumeric + dashes
        if clean in TRIM_TOKENS:
            continue
        if len(clean) <= 1 and clean.isalpha():  # lone letters are trim markers
            continue
        base.append(clean)

    result = ' '.join(base).strip()
    # Fallback: return first token if everything was stripped
    return result if result else name.split()[0]


_PUNCT_RE = re.compile(r'[\s\-_/]')


def _similarity(a: str, b: str) -> int:
    """
    Fuzzy similarity between two stripped model names.

    Special cases:
    - Punctuation-only differences ('cr-v' vs 'crv', 'f-150' vs 'f150') always
      score 100 so they merge regardless of threshold.
    - Names that both end in 'class' ('c-class', 'e-class', 's-class') are
      never merged with each other — they are distinct Mercedes model lines.
    - Names containing different leading numbers ('silverado 1500' vs
      'silverado 2500hd') are never merged.
    """
    # Fast path: identical after stripping punctuation/spaces → definitely same model
    if _PUNCT_RE.sub('', a) == _PUNCT_RE.sub('', b):
        return 100

    # Guard: Mercedes-style "{X}-class" names must not merge with each other
    if a.endswith('class') and b.endswith('class'):
        return 0

    # Guard: different leading numeric identifiers mean different model tiers
    nums_a = _NUM_RE.findall(a)
    nums_b = _NUM_RE.findall(b)
    if nums_a and nums_b and fuzz.ratio(nums_a[0], nums_b[0]) < 80:
        return 0

    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b))


def build_model_map(model_series: pd.Series, threshold: int = 82) -> dict:
    """
    Given a Series of raw model strings for one make, return a dict mapping
    every raw model string to its canonical base model name.

    Steps:
      1. Strip trim tokens from every unique raw model string.
      2. Greedy-cluster stripped names by fuzzy similarity — the most frequent
         name becomes the canonical for its cluster.
      3. Prefix pass — any remaining cluster head whose stripped name starts
         with a known canonical gets absorbed into it. This catches the
         Craigslist free-text noise ("civic 5-speed", "cr-v 1 owner", etc.)
         that survives trim stripping but begins with a valid base model.
    """
    unique_raws = [m for m in model_series.dropna().unique() if str(m) not in ('nan', '')]
    if not unique_raws:
        return {}

    stripped_of = {raw: strip_trims(raw) for raw in unique_raws}

    stripped_series = model_series.map(lambda x: stripped_of.get(x, ''))
    counts = stripped_series.value_counts()
    ordered = [n for n in counts.index if n]   # most frequent first

    # Pass 1 — greedy fuzzy clustering
    cluster_head: dict[str, str] = {}
    for name in ordered:
        if name in cluster_head:
            continue
        cluster_head[name] = name
        for other in ordered:
            if other not in cluster_head and _similarity(name, other) >= threshold:
                cluster_head[other] = name

    # Pass 2 — prefix absorption
    # Any cluster head whose stripped name starts with another canonical (+ space)
    # gets absorbed — UNLESS the trailing suffix contains a standalone 3-digit-plus
    # number, which would indicate a model tier ("silverado 1500" vs "silverado",
    # "sierra 2500hd" vs "sierra").
    _BIG_NUM = re.compile(r'\d{3,}')
    canonicals = sorted(
        {v for v in cluster_head.values() if cluster_head.get(v) == v},
        key=len, reverse=True,   # longest first so "grand cherokee" beats "grand"
    )
    for name in ordered:
        if cluster_head.get(name) != name:
            continue   # already absorbed
        for canon in canonicals:
            if canon == name:
                continue
            if name.startswith(canon + ' '):
                extra = name[len(canon):].strip()
                if _BIG_NUM.search(extra):
                    continue   # suffix has a model-tier number — keep separate
                cluster_head[name] = canon
                break

    def _resolve(name: str) -> str:
        """Follow the cluster_head chain to its ultimate canonical."""
        seen: set[str] = set()
        while name in cluster_head and cluster_head[name] != name:
            if name in seen:
                break
            seen.add(name)
            name = cluster_head[name]
        return cluster_head.get(name, name)

    return {raw: _resolve(stripped_of.get(raw, '')) for raw in unique_raws}
