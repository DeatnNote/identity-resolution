"""Standardisation: the cheap, deterministic cleaning that makes matching possible."""
import re

import jellyfish

from .generate import FIRST

# nickname -> canonical first name (ambiguous nicknames map to the first canonical name seen)
NICK = {}
for canon, nicks in FIRST.items():
    NICK.setdefault(canon, canon)
    for n in nicks:
        NICK.setdefault(n, canon)
STREET_ABBR = {"street": "st", "avenue": "ave", "road": "rd", "drive": "dr", "lane": "ln",
               "court": "ct", "place": "pl", "boulevard": "blvd"}


def clean_email(e):
    if not isinstance(e, str) or "@" not in e:
        return None
    local, domain = e.strip().lower().split("@", 1)
    local = local.split("+", 1)[0]                 # drop +tags
    if domain in ("gmail.com", "googlemail.com"):
        local = local.replace(".", "")             # gmail ignores dots
        domain = "gmail.com"
    return f"{local}@{domain}"


def clean_phone(p):
    if not isinstance(p, str):
        return None
    d = re.sub(r"\D", "", p)
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    return d if len(d) == 10 else None


def clean_address(a):
    if not isinstance(a, str):
        return None
    toks = re.sub(r"[^\w\s]", " ", a.lower()).split()
    return " ".join(STREET_ABBR.get(t, t) for t in toks) or None


def split_name(n):
    toks = re.sub(r"[^\w\s]", " ", str(n).lower()).split()
    if not toks:
        return None, None
    if len(toks) == 1:
        return None, toks[0]
    return toks[0], toks[-1]


def standardize(df):
    out = df.copy()
    names = out["full_name"].map(split_name)
    out["first_raw"] = names.str[0]
    out["last_name"] = names.str[1]
    out["first_name"] = out["first_raw"].map(lambda f: NICK.get(f, f) if isinstance(f, str) else None)
    out["first_initial"] = out["first_raw"].str[0]
    out["last_soundex"] = out["last_name"].map(lambda s: jellyfish.soundex(s) if isinstance(s, str) else None)
    out["email_clean"] = out["email"].map(clean_email)
    out["phone_clean"] = out["phone"].map(clean_phone)
    out["address_clean"] = out["address"].map(clean_address)
    out["zip5"] = out["zip"].astype(str).str.extract(r"(\d{5})")[0]
    return out
