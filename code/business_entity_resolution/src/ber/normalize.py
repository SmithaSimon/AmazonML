"""Text normalisation for business names and addresses.

Everything here is language/country agnostic: accents are folded, non-Latin
scripts are romanised (unidecode) and then "squashed" so that romanised
Indic text looks closer to its English spelling, generic abbreviations are
expanded, and legal suffixes are separated from the core name.
"""
from __future__ import annotations

import re
from typing import Iterable

import polars as pl
from unidecode import unidecode

# --------------------------------------------------------------------------
# Legal suffixes / boilerplate tokens that carry no identity information.
# Multi-country on purpose (US, India, France, generic European).
# --------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp",
    "lp", "ltd", "limited", "pvt", "private", "plc", "pc", "pllc", "pa",
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "eirl", "scop",
    "ag", "gmbh", "bv", "nv", "ltda", "srl", "spa",
    "m/s", "ms", "mr", "mrs", "dba", "the", "and", "of", "de", "du", "des", "la", "le", "les", "et",
}

# Generic abbreviation expansion applied to both names and addresses.
# Keys are whole lowercase tokens.  Values may contain spaces.
ABBREV = {
    # street types (EN)
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place",
    "hwy": "highway", "pkwy": "parkway", "sq": "square", "ter": "terrace", "trl": "trail",
    "pt": "point", "mt": "mount", "ft": "fort", "hts": "heights", "pk": "park",
    "twp": "township", "cty": "county", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "fl": "floor", "ste": "suite", "apt": "apartment", "bldg": "building", "rm": "room",
    "po": "po", "pmb": "pmb",
    # street types (FR)
    "r": "rue", "bd": "boulevard", "all": "allee", "ch": "chemin", "imp": "impasse",
    "pl.": "place", "crs": "cours", "qu": "quai", "rte": "route", "sq.": "square",
    "gal": "galerie", "res": "residence", "lot": "lotissement", "zi": "zone industrielle",
    "za": "zone artisanale", "zac": "zone amenagement",
    # India
    "nr": "near", "opp": "opposite", "h": "house", "hno": "house no", "vill": "village",
    "dist": "district", "tq": "taluka", "tal": "taluka", "teh": "tehsil", "mkt": "market",
    "soc": "society", "colny": "colony", "apts": "apartments", "bldg.": "building",
    "ind": "industrial", "indl": "industrial", "estt": "estate", "sec": "sector", "ph": "phase",
    "kh": "khasra", "gali": "gali", "mg": "mg", "nagar": "nagar",
    # name abbreviations
    "intl": "international", "natl": "national", "mfg": "manufacturing", "svc": "service",
    "svcs": "services", "assoc": "associates", "assocs": "associates", "bros": "brothers",
    "tech": "technologies", "ent": "enterprises", "ents": "enterprises", "ind.": "industries",
    "engg": "engineering", "constr": "construction", "dev": "development", "mgmt": "management",
    "sol": "solutions", "sols": "solutions", "sys": "systems", "univ": "university",
    "hosp": "hospital", "dept": "department", "ctr": "center", "centre": "center",
    "ste.": "sainte", "st.": "saint", "gen": "general", "genl": "general",
}

# Tokens that are pure "address type" words: useful for parsing, useless as blocking keys.
ADDR_STOP = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "court", "circle", "place",
    "highway", "parkway", "square", "terrace", "trail", "point", "floor", "suite", "apartment",
    "building", "room", "unit", "no", "number", "house", "near", "opposite", "village",
    "district", "taluka", "tehsil", "market", "society", "colony", "sector", "phase",
    "rue", "allee", "chemin", "impasse", "cours", "quai", "route", "residence", "lotissement",
    "zone", "industrielle", "artisanale", "amenagement", "north", "south", "east", "west",
    "po", "box", "pmb", "the", "of", "and", "de", "du", "des", "la", "le", "les", "at", "a",
    "block", "plot", "flat", "shop", "gat", "khasra", "main", "cross", "nagar", "town",
    "city", "township", "county", "dist", "area", "layout", "estate", "industrial",
}

_DOMAIN_RE = re.compile(
    r"^(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|in|co\.in|co|io|fr|biz|info|us|eu|co\.uk)$"
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
_NUM_RE = re.compile(r"\d+")
_LEADNUM_RE = re.compile(r"^\D{0,4}?(\d+)")


def squash_romanised(s: str) -> str:
    """Make unidecode's Indic romanisation look more like English spelling.

    e.g. 'praaivett limittedd' -> 'praivet limited', 'iNddo' -> 'indo'.
    """
    s = s.replace("N", "n").replace("'", "")
    s = re.sub(r"aa", "a", s)
    s = re.sub(r"ii", "i", s)
    s = re.sub(r"uu", "u", s)
    s = re.sub(r"ee", "e", s)
    s = re.sub(r"oo", "o", s)
    s = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1", r"\1", s)  # tt->t, dd->d ...
    return s


def romanise(s: str) -> str:
    """Fold accents and transliterate non-Latin scripts; ASCII passes through untouched."""
    if s.isascii():
        return s
    out = unidecode(s)
    # unidecode output for Indic scripts contains capital N and doubled letters.
    if _NON_ASCII_RE.search(s) and re.search(r"[ऀ-෿]", s):
        out = squash_romanised(out)
    return out


def _expand_tokens(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        rep = ABBREV.get(t)
        if rep is None:
            out.append(t)
        else:
            out.extend(rep.split())
    return out


_PRE_RE = re.compile(r"(?i)\bn\s*[°º]|\bm/s\b|[°º]")


def basic_tokens(s: str) -> list[str]:
    """Romanise, lowercase, & -> and, strip punctuation, expand abbreviations."""
    s = _PRE_RE.sub(" ", s)
    s = romanise(s).lower().replace("&", " and ").replace("+", " plus ")
    s = _NON_ALNUM_RE.sub(" ", s)
    toks = [(t.lstrip("0") or "0") if t.isdigit() else t for t in s.split()]
    return _expand_tokens(toks)


def parse_name(raw: str) -> dict:
    """Return name_norm, name_core (suffix stripped), is_domain, domain_stem, nonlatin flag."""
    raw_l = raw.strip().lower()
    m = _DOMAIN_RE.match(raw_l.replace(" ", ""))
    nonlatin = int(bool(re.search(r"[ऀ-෿؀-ۿ一-鿿]", raw)))
    if m:
        stem = m.group(1).replace("-", "")
        return {
            "name_norm": stem,
            "name_core": stem,
            "name_nospace": stem,
            "is_domain": 1,
            "name_nonlatin": nonlatin,
        }
    toks = basic_tokens(raw)
    core = [t for t in toks if t not in LEGAL_SUFFIXES]
    if not core:  # name was only suffixes; keep everything
        core = toks
    return {
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_nospace": "".join(core),
        "is_domain": 0,
        "name_nonlatin": nonlatin,
    }


def parse_address(raw: str) -> dict:
    """Return addr_norm, addr_words (non-stop alpha tokens), nums, house_num."""
    if not raw or not raw.strip():
        return {"addr_norm": "", "addr_words": "", "addr_nums": "", "house_num": "", "addr_empty": 1}
    # House number: leading number of the first comma-component that starts with a digit
    # (components may be reordered, so scan all of them).
    house = ""
    for comp in raw.split(","):
        c = comp.strip().lower()
        c = re.sub(r"^(?:[a-z]{1,2}\.?\s*)?(?:no|nos|n\s*[°º]|#|number)\.?\s*[-:#.]*\s*", "", c)
        m = _LEADNUM_RE.match(c)
        if m and c[: m.start(1)].strip(" -#.") == "":
            house = m.group(1).lstrip("0") or "0"
            break
    toks = basic_tokens(raw)
    nums = sorted({t.lstrip("0") or "0" for t in toks if t.isdigit()})
    words = [t for t in toks if t.isalpha() and t not in ADDR_STOP and len(t) > 1]
    return {
        "addr_norm": " ".join(toks),
        "addr_words": " ".join(words),
        "addr_nums": " ".join(nums),
        "house_num": house,
        "addr_empty": 0,
    }


def normalise_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Add normalised columns to a records frame (entity_id, business_name, business_address, country)."""
    names = [parse_name(x) for x in df["business_name"].to_list()]
    addrs = [parse_address(x) for x in df["business_address"].to_list()]
    nf = pl.DataFrame(names)
    af = pl.DataFrame(addrs)
    out = pl.concat([df, nf, af], how="horizontal")
    out = out.with_columns(
        pl.col("entity_id").str.slice(0, 2).alias("src"),
        pl.col("is_domain").cast(pl.Int8),
        pl.col("name_nonlatin").cast(pl.Int8),
        pl.col("addr_empty").cast(pl.Int8),
    )
    return out
