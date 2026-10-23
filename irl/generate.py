"""
Synthetic restaurant-customer records for identity resolution.

Each synthetic PERSON appears as 1-6 RECORDS across three source systems:

  pos      point-of-sale: name often abbreviated ("J Smith"), card last-4, zip; rarely email
  online   online ordering: email, phone, delivery address
  loyalty  loyalty programme: full name, email, phone, address, birth year

Realistic problems are injected on purpose:
  * nicknames (Robert -> Bob), typos, swapped first/last, upper/lower case noise
  * email variants (gmail dots, +tags, a second address), phone formatting and wrong digits
  * missing fields that depend on the source, address abbreviations, people who MOVED
  * households: family members share last name, address and sometimes phone (hard negatives)
  * Zipf-distributed surnames and zip codes, so some blocks are huge (the skew a Spark job hits)

Everything is generated; no real person's data is used. Names come from short public
lists of common US first and last names.
"""
import numpy as np
import pandas as pd

FIRST = {
    "james": ["jim", "jimmy"], "robert": ["bob", "rob", "bobby"], "john": ["johnny", "jack"],
    "michael": ["mike", "mikey"], "william": ["bill", "will", "billy"], "david": ["dave"],
    "richard": ["rick", "dick", "rich"], "joseph": ["joe", "joey"], "thomas": ["tom", "tommy"],
    "charles": ["charlie", "chuck"], "christopher": ["chris"], "daniel": ["dan", "danny"],
    "matthew": ["matt"], "anthony": ["tony"], "steven": ["steve"], "andrew": ["andy", "drew"],
    "joshua": ["josh"], "kenneth": ["ken", "kenny"], "kevin": [], "brian": [], "george": [],
    "edward": ["ed", "eddie", "ted"], "ronald": ["ron", "ronnie"], "timothy": ["tim"],
    "jason": [], "jeffrey": ["jeff"], "ryan": [], "jacob": ["jake"], "gary": [], "nicholas": ["nick"],
    "eric": [], "jonathan": ["jon"], "stephen": ["steve"], "larry": [], "justin": [], "scott": [],
    "benjamin": ["ben"], "samuel": ["sam"], "gregory": ["greg"], "alexander": ["alex"],
    "patrick": ["pat"], "raymond": ["ray"], "jack": [], "dennis": [], "jerry": [], "tyler": [],
    "aaron": [], "jose": [], "adam": [], "nathan": ["nate"], "henry": ["hank"], "peter": ["pete"],
    "mary": [], "patricia": ["pat", "patty", "trish"], "jennifer": ["jen", "jenny"],
    "linda": [], "elizabeth": ["liz", "beth", "lizzie"], "barbara": ["barb"], "susan": ["sue"],
    "jessica": ["jess"], "sarah": [], "karen": [], "lisa": [], "nancy": [], "betty": [],
    "margaret": ["maggie", "peggy"], "sandra": ["sandy"], "ashley": [], "kimberly": ["kim"],
    "emily": [], "donna": [], "michelle": [], "carol": [], "amanda": ["mandy"], "dorothy": ["dot"],
    "melissa": ["missy"], "deborah": ["deb", "debbie"], "stephanie": ["steph"], "rebecca": ["becky"],
    "sharon": [], "laura": [], "cynthia": ["cindy"], "kathleen": ["kathy", "kate"], "amy": [],
    "angela": ["angie"], "shirley": [], "anna": [], "brenda": [], "pamela": ["pam"], "emma": [],
    "nicole": ["nikki"], "helen": [], "samantha": ["sam"], "katherine": ["kate", "katie", "kathy"],
    "christine": ["chris"], "debra": ["deb"], "rachel": [], "carolyn": [], "janet": [],
    "catherine": ["cathy", "kate"], "maria": [], "heather": [], "diane": [], "ruth": [], "julie": [],
    "olivia": ["liv"], "joyce": [], "virginia": ["ginny"], "victoria": ["vicky", "tori"],
    "kelly": [], "lauren": [], "christina": ["tina"], "joan": [], "evelyn": [], "judith": ["judy"],
    "megan": [], "andrea": [], "cheryl": [], "hannah": [], "jacqueline": ["jackie"], "martha": [],
    "gloria": [], "teresa": ["terry"], "ann": [], "sara": [], "madison": ["maddie"], "frances": ["fran"],
    "kathryn": ["kate"], "janice": [], "jean": [], "abigail": ["abby"], "alice": [], "julia": [],
    "judy": [], "sophia": ["sophie"], "grace": [], "denise": [], "amber": [], "doris": [], "marilyn": [],
    "danielle": ["dani"], "beverly": ["bev"], "isabella": ["bella"], "theresa": ["terry"], "diana": [],
    "natalie": [], "brittany": [], "charlotte": [], "marie": [], "kayla": [], "alexis": [], "lori": [],
}
LAST = ["smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis", "rodriguez",
        "martinez", "hernandez", "lopez", "gonzalez", "wilson", "anderson", "thomas", "taylor", "moore",
        "jackson", "martin", "lee", "perez", "thompson", "white", "harris", "sanchez", "clark",
        "ramirez", "lewis", "robinson", "walker", "young", "allen", "king", "wright", "scott", "torres",
        "nguyen", "hill", "flores", "green", "adams", "nelson", "baker", "hall", "rivera", "campbell",
        "mitchell", "carter", "roberts", "gomez", "phillips", "evans", "turner", "diaz", "parker",
        "cruz", "edwards", "collins", "reyes", "stewart", "morris", "morales", "murphy", "cook",
        "rogers", "gutierrez", "ortiz", "morgan", "cooper", "peterson", "bailey", "reed", "kelly",
        "howard", "ramos", "kim", "cox", "ward", "richardson", "watson", "brooks", "chavez", "wood",
        "james", "bennett", "gray", "mendoza", "ruiz", "hughes", "price", "alvarez", "castillo",
        "sanders", "patel", "myers", "long", "ross", "foster", "jimenez", "powell", "jenkins", "perry",
        "russell", "sullivan", "bell", "coleman", "butler", "henderson", "barnes", "gonzales",
        "fisher", "vasquez", "simmons", "romero", "jordan", "patterson", "alexander", "hamilton",
        "graham", "reynolds", "griffin", "wallace", "moreno", "west", "cole", "hayes", "bryant",
        "herrera", "gibson", "ellis", "tran", "medina", "aguilar", "stevens", "murray", "ford",
        "castro", "marshall", "owens", "harrison", "fernandez", "mcdonald", "woods", "washington",
        "kennedy", "wells", "vargas", "henry", "chen", "freeman", "webb", "tucker", "guzman", "burns",
        "crawford", "olson", "simpson", "porter", "hunter", "gordon", "mendez", "silva", "shaw",
        "snyder", "mason", "dixon", "munoz", "hunt", "hicks", "holmes", "palmer", "wagner", "black",
        "robertson", "boyd", "rose", "stone", "salazar", "fox", "warren", "mills", "meyer", "rice",
        "schmidt", "garza", "daniels", "ferguson", "nichols", "stephens", "soto", "weaver", "ryan",
        "gardner", "payne", "grant", "dunn", "kelley", "spencer", "hawkins", "arnold", "pierce"]
STREETS = ["main", "oak", "pine", "maple", "cedar", "elm", "washington", "lake", "hill", "park",
           "walnut", "sunset", "lincoln", "jackson", "church", "river", "highland", "mill", "spring",
           "ridge", "chestnut", "franklin", "center", "forest", "valley", "meadow", "broad", "union"]
STYPES = [("street", "st"), ("avenue", "ave"), ("road", "rd"), ("drive", "dr"), ("lane", "ln"),
          ("court", "ct"), ("place", "pl"), ("boulevard", "blvd")]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "aol.com"]
SOURCES = {"pos": 0.45, "online": 0.30, "loyalty": 0.25}


def _zipf_choice(rng, items, n, a=1.1):
    w = 1.0 / np.arange(1, len(items) + 1) ** a
    return rng.choice(items, n, p=w / w.sum())


def _typo(s, rng):
    if len(s) < 3:
        return s
    i = int(rng.integers(1, len(s) - 1))
    op = rng.integers(0, 3)
    if op == 0:
        return s[:i] + s[i + 1:]                                   # deletion
    if op == 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]                 # transposition
    return s[:i] + "aeiourstln"[int(rng.integers(0, 10))] + s[i + 1:]   # substitution


def make_people(n, rng):
    firsts = np.array(list(FIRST))
    zips = np.array([f"{z:05d}" for z in rng.choice(np.arange(10001, 99950), 400, replace=False)])
    people = pd.DataFrame({
        "person_id": np.arange(n),
        "first": rng.choice(firsts, n),
        "last": _zipf_choice(rng, LAST, n, 0.9),
        "zip": _zipf_choice(rng, zips, n, 0.8),
        "birth_year": rng.integers(1945, 2006, n),
    })
    people["house_no"] = rng.integers(1, 9999, n)
    people["street"] = rng.choice(STREETS, n)
    people["stype"] = rng.integers(0, len(STYPES), n)
    people["phone"] = [f"{rng.integers(201, 990)}{rng.integers(200, 999)}{rng.integers(0, 10000):04d}" for _ in range(n)]
    people["card4"] = [f"{rng.integers(0, 10000):04d}" for _ in range(n)]
    # households: ~25% of people join another person's household (same last name + address)
    hh = rng.random(n) < 0.25
    hosts = rng.integers(0, n, n)
    for c in ("last", "zip", "house_no", "street", "stype"):
        people.loc[hh, c] = people.loc[hosts[hh], c].to_numpy()
    share_phone = hh & (rng.random(n) < 0.4)                       # landline shared at home
    people.loc[share_phone, "phone"] = people.loc[hosts[share_phone], "phone"].to_numpy()
    people["household_of"] = np.where(hh, hosts, people["person_id"])
    # email
    style = rng.integers(0, 4, n)
    base = np.where(style == 0, people["first"] + "." + people["last"],
           np.where(style == 1, people["first"].str[0] + people["last"],
           np.where(style == 2, people["first"] + people["last"] + people["birth_year"].astype(str).str[-2:],
                    people["last"] + people["first"].str[0] + rng.integers(1, 99, n).astype(str))))
    people["email"] = base + "@" + rng.choice(DOMAINS, n)
    # ~6% of people move during the period: a second address
    people["moved"] = rng.random(n) < 0.06
    people["new_house_no"] = rng.integers(1, 9999, n)
    people["new_street"] = rng.choice(STREETS, n)
    return people


def _address(p, rng, moved):
    no = p["new_house_no"] if moved else p["house_no"]
    st = p["new_street"] if moved else p["street"]
    full, abbr = STYPES[int(p["stype"])]
    t = abbr if rng.random() < 0.5 else full
    return f"{no} {st} {t}"


def make_records(people, rng, mean_records=2.5, source_weights=None, noise=1.0):
    sw = source_weights or SOURCES
    srcs, sp = list(sw), np.array(list(sw.values()))
    n_rec = 1 + rng.poisson(mean_records - 1, len(people))
    n_rec = np.minimum(n_rec, 6)
    rows = []
    for p, k in zip(people.itertuples(index=False), n_rec):
        p = p._asdict()
        for _ in range(int(k)):
            src = rng.choice(srcs, p=sp / sp.sum())
            first, last = p["first"], p["last"]
            if FIRST[first] and rng.random() < 0.15 * noise:
                first = rng.choice(FIRST[first])
            if rng.random() < 0.06 * noise:
                first = _typo(first, rng)
            if rng.random() < 0.06 * noise:
                last = _typo(last, rng)
            if rng.random() < 0.01 * noise:
                first, last = last, first
            email = p["email"]
            r = rng.random()
            if r < 0.05 * noise and email.endswith("gmail.com"):
                local = email.split("@")[0]
                email = local.replace(".", "") + "@gmail.com" if "." in local else local[:2] + "." + local[2:] + "@gmail.com"
            elif r < 0.09 * noise:
                email = email.split("@")[0] + f"+{rng.choice(['orders', 'promo', 'food'])}@" + email.split("@")[1]
            elif r < 0.13 * noise:
                email = email.split("@")[0] + "@" + rng.choice(DOMAINS)       # a different mailbox
            phone = p["phone"]
            if rng.random() < 0.03 * noise:
                i = int(rng.integers(3, 10))
                phone = phone[:i] + str(int(rng.integers(0, 10))) + phone[i + 1:]
            fmt = rng.integers(0, 4)
            phone = [phone, f"({phone[:3]}) {phone[3:6]}-{phone[6:]}", f"{phone[:3]}-{phone[3:6]}-{phone[6:]}",
                     f"+1{phone}"][fmt]
            moved = p["moved"] and rng.random() < 0.5
            addr = _address(p, rng, moved)
            name = f"{first} {last}"
            if src == "pos" and rng.random() < 0.5:
                name = f"{first[0]} {last}"
            if rng.random() < 0.3:
                name = name.upper()
            rec = {"source": src, "full_name": name, "email": email, "phone": phone, "address": addr,
                   "zip": p["zip"], "card_last4": p["card4"], "birth_year": p["birth_year"],
                   "person_id": p["person_id"], "household_of": p["household_of"]}
            # source-dependent missingness
            if src == "pos":
                rec["email"] = rec["email"] if rng.random() < 0.10 else None
                rec["phone"] = rec["phone"] if rng.random() < 0.30 else None
                rec["address"] = None
                rec["birth_year"] = None
            elif src == "online":
                rec["card_last4"] = rec["card_last4"] if rng.random() < 0.6 else None
                rec["birth_year"] = None
                rec["phone"] = rec["phone"] if rng.random() < 0.85 else None
            else:
                rec["card_last4"] = None
                rec["email"] = rec["email"] if rng.random() < 0.95 else None
            rows.append(rec)
    df = pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(0, 1 << 31))).reset_index(drop=True)
    df.insert(0, "record_id", [f"R{i:07d}" for i in range(len(df))])
    return df


def generate(n_people=20000, seed=0):
    """Return (records with truth columns, people)."""
    rng = np.random.default_rng(seed)
    people = make_people(n_people, rng)
    return make_records(people, rng), people


def drifted_batch(people, seed=1):
    """A later batch where a new 'kiosk' source arrives with different habits (for drift demos)."""
    rng = np.random.default_rng(seed)
    sub = people.sample(frac=0.25, random_state=seed)
    recs = make_records(sub, rng, mean_records=1.6,
                        source_weights={"pos": 0.3, "online": 0.2, "loyalty": 0.1, "kiosk": 0.4}, noise=1.6)
    kiosk = recs["source"] == "kiosk"
    recs.loc[kiosk, "email"] = None
    recs.loc[kiosk & (rng.random(len(recs)) < 0.7), "phone"] = None
    recs.loc[kiosk, "full_name"] = recs.loc[kiosk, "full_name"].str.split().str[-1].str.upper()  # last name only
    recs["record_id"] = "K" + recs["record_id"].str[1:]
    return recs
