"""Entity vocabularies, span extraction, and same-kind variant generation.

Design rules:
- The abandoned-value pool IS the kept-value pool (anti-association): every
  enumerated entry can appear both as a correct value in clean outputs and as
  an abandoned value in corrections. No word is ever "the correction word".
- Extraction is deterministic and metadata-driven (no rng): used on clean,
  well-formed outputs to derive entity metadata, and by validators/metrics.
- entity_variant() always returns a same-kind value, never equal to the input,
  preserving the format family (digits stay digits, prices stay prices, etc.).
"""

import random
import re
from dataclasses import dataclass

from data.common import normalize

ENTITY_KINDS = ("name", "item", "location", "tech", "number", "time", "date")


class SlotError(Exception):
    """Raised when a plan requires an entity kind that a text does not contain."""


# --------------------------------------------------------------------------
# Enumerated pools
# --------------------------------------------------------------------------

FIRST_NAMES = (
    "Sarah", "John", "Omar", "Priya", "Alex", "Maria", "Ahmed", "Emma", "Liam", "Noah",
    "Olivia", "Sofia", "Ethan", "Aisha", "David", "Lena", "Yusuf", "Clara", "Daniel", "Hana",
    "Karim", "Grace", "Hassan", "Julia", "Adam", "Nora", "Samir", "Chloe", "Peter", "Zara",
    "Mark", "Fatima", "Lucas", "Amara", "Nadia", "Oscar", "Layla", "Victor", "Rania", "Gabriel",
    "Ines", "Tariq", "Elena", "Sam", "Maya", "Tom", "Hannah", "Ryan", "Leila", "Chris",
    "Diana", "Jamal", "Eva", "Michael", "Salma", "George", "Rina", "Khalid", "Anna", "Felix",
    "Nour", "Henry", "Imani", "Jack", "Dalia", "Ben", "Aya", "Diego", "Mona", "Paul",
    "Yara", "Tarek", "Rosa", "Will", "Nadine", "Hasan", "Alina", "James", "Meera", "Tomas",
    "Ruben", "Sana", "Leo", "Camila", "Ravi", "Isabelle", "Farah", "Max", "Gita", "Bruno",
    "Selin", "Pedro", "Aaliyah", "Ivan", "Esra", "Malik", "Wendy", "Faisal", "Lina", "Andre",
    "Petra", "Asha", "Nathan", "Carmen", "Bilal", "Serena", "Dario", "Amani", "Kyle", "Martina",
    "Reem", "Hugo", "Tara", "Kenji", "Alma", "Rohan", "Vera", "Marcel", "Kira", "Arjun",
    "Dana", "Oskar", "Anya", "Rashid", "Beatriz", "Viktor", "Samra", "Ehsan", "Giulia", "Renata",
    "Fahad", "Celeste", "Timur", "Wafaa", "Dennis", "Alba", "Kunal", "Miriam", "Stefan", "Priyanka",
    "Youssef", "Amira", "Cedric", "Lotte", "Danish", "Roshan", "Sibel", "Emre", "Vicky", "Rahim",
)

LAST_NAMES = (
    "Chen", "Patel", "Smith", "Garcia", "Kim", "Ali", "Johnson", "Nguyen", "Brown", "Khan",
    "Martinez", "Lee", "Haddad", "Rossi", "Muller", "Silva", "Okafor", "Tanaka", "Novak", "Rahman",
    "Santos", "Ibrahim", "Zhang", "Dubois", "Kowalski", "Fernandez", "Suzuki", "Costa", "Petrova",
    "Al-Farsi", "Weber", "Johansson", "Moreau", "Bianchi", "Osei", "Bautista", "Rashid", "Lindqvist",
    "Marchetti", "Fonseca", "Hamdan", "Svensson", "Popescu", "Adeyemi", "Valdez", "Kaur", "Nakagawa",
    "Dimitrov", "O'Brien", "van der Berg",
)

# Bare nouns (no articles): templates supply "the {item}" / "some {item}" as needed.
# The pool is partitioned by grammatical number so agreement-sensitive slots
# ({item_s} / {item_p}) can never produce "the curtains needs updating".
SINGULAR_ITEMS = (
    "milk", "bread", "paratha", "coffee", "tea", "sugar", "flour", "rice", "pasta",
    "olive oil", "butter", "cheese", "yogurt", "chicken", "salmon", "garlic", "ginger",
    "lettuce", "spinach", "cumin", "cinnamon", "salt", "pepper", "honey", "oatmeal",
    "cereal", "orange juice", "mineral water", "chocolate", "soap", "shampoo",
    "toothpaste", "detergent", "notebook", "printer paper", "thumb drive", "HDMI cable",
    "power adapter", "microphone", "webcam", "mouse", "keyboard", "monitor stand",
    "phone charger", "laptop bag", "backpack", "suitcase", "umbrella", "coat", "scarf",
    "jacket", "yoga mat", "water bottle", "sunscreen", "first aid kit", "hand sanitizer",
    "dish soap", "aluminum foil", "plastic wrap", "dog food", "cat litter",
    "plant fertilizer", "potting soil", "paint", "duct tape", "extension cord",
    "engine oil", "windshield fluid", "air filter", "tool kit", "measuring tape",
    "rope", "hammer", "ladder", "watering can", "flashlight", "tent", "sleeping bag",
    "cooler", "picnic blanket", "beach towel", "insect repellent", "planner", "calendar",
    "desk lamp", "fan", "heater", "kettle", "toaster", "blender", "rice cooker",
    "frying pan", "cutting board", "whisk", "dish rack", "fabric softener", "iron",
    "sewing kit", "hairdryer", "razor", "shaving cream", "toothbrush", "floss", "lotion",
    "mirror", "clock", "rug", "lamp shade", "pillow", "blanket", "mop", "broom",
    "vacuum bag", "key chain", "wallet", "belt", "hat", "dress", "blouse", "suit",
    "tie", "swimsuit", "raincoat", "bike lock", "helmet", "skateboard", "football",
    "basketball", "badminton set", "chess set", "puzzle", "board game", "book",
    "magazine", "newspaper", "pen holder", "whiteboard", "stapler", "hole punch",
    "shredder", "projector", "printer", "router", "modem", "ethernet cable", "SIM card",
    "memory card", "camera", "drone", "smart speaker", "smartwatch", "tablet",
    "power bank", "laptop", "invoice", "spreadsheet", "contract", "proposal",
)

PLURAL_ITEMS = (
    "eggs", "apples", "bananas", "oranges", "strawberries", "avocados", "tomatoes",
    "onions", "potatoes", "biscuits", "paper towels", "batteries", "light bulbs",
    "envelopes", "stamps", "pens", "folders", "binder clips", "sticky notes", "cables",
    "headphones", "passport photos", "gloves", "sneakers", "socks", "dress shirts",
    "jeans", "running shoes", "dumbbells", "painkillers", "vitamins", "face masks",
    "tissues", "trash bags", "sponges", "napkins", "coffee filters", "tea bags",
    "paint brushes", "screws", "drill bits", "candles", "matches", "nails",
    "garden gloves", "seeds", "ice packs", "kitchen knives", "measuring cups",
    "oven mitts", "laundry pods", "curtains", "sheets", "towels", "sunglasses",
    "boots", "sandals", "tennis balls", "playing cards", "markers", "paper clips",
    "ink cartridges", "project files", "slides", "scissors",
)

ITEMS = SINGULAR_ITEMS + PLURAL_ITEMS


LOCATIONS = (
    "Dubai", "Abu Dhabi", "Berlin", "London", "Paris", "Tokyo", "Mumbai", "Delhi", "New York",
    "Chicago", "Seattle", "Austin", "Toronto", "Vancouver", "Singapore", "Sydney", "Melbourne",
    "Cairo", "Istanbul", "Madrid", "Barcelona", "Lisbon", "Rome", "Milan", "Vienna", "Prague",
    "Warsaw", "Amsterdam", "Brussels", "Zurich", "Geneva", "Stockholm", "Oslo", "Helsinki",
    "Copenhagen", "Dublin", "Glasgow", "Manchester", "Boston", "Denver", "Miami", "Atlanta",
    "Houston", "Portland", "San Francisco", "Los Angeles", "Nairobi", "Lagos", "Accra",
    "Casablanca", "Beirut", "Amman", "Doha", "Riyadh", "Karachi", "Lahore", "Dhaka", "Colombo",
    "Bangkok", "Jakarta", "Manila", "Seoul", "Osaka", "Shanghai", "Beijing", "Athens", "Budapest",
    "Bucharest", "Belgrade", "Riga", "Tallinn", "Vilnius", "the office", "the cafe", "the airport",
    "the hotel", "the lab", "the warehouse", "the studio", "the clinic",
    "the campus", "the station", "Room 302", "Building B", "the third floor", "the lobby",

    "the conference room", "the server room", "the garage", "the garden", "the beach", "the mall",
    "the market", "the pharmacy", "the bank", "the post office",
)

# Bare phrases (no leading "the"): templates always write "the {tech}".
TECH_TERMS = (
    "Kubernetes", "Docker", "Postgres", "MongoDB", "Redis", "Kafka", "Elasticsearch", "Grafana",
    "Prometheus", "Terraform", "Ansible", "Jenkins", "GitLab", "GitHub", "Nginx", "Apache",
    "Node.js", "React", "Flutter", "Python", "TypeScript", "JavaScript", "Rust", "Golang", "Java",
    "C++", "SQL", "GraphQL", "REST API", "WebSocket", "gRPC", "OAuth", "JWT", "TLS", "SSH", "VPN",
    "DNS", "HTTP", "HTTPS", "IPv6", "firewall", "load balancer", "CI pipeline",
    "staging cluster", "production server", "database", "cache layer",
    "message queue", "backup script", "deployment script", "cron job",
    "API keys", "SSH keys", "log files", "config file", "Docker image",
    "codebase", "repository", "branch", "pull request", "test suite",
    "lint step", "build job", "rollout", "hotfix", "release notes",
    "changelog", "migration", "schema", "query", "endpoint", "webhook",
    "service worker", "sandbox", "staging environment", "dev server", "localhost",
    "IDE", "terminal", "command line", "compiler", "debugger", "profiler",
    "emulator", "simulator", "virtual machine", "container", "cluster",
    "node pool", "ingress", "cert", "domain", "subdomain", "dashboard",
    "monitor", "alert", "incident", "uptime", "SLA", "sprint",
    "backlog", "roadmap", "milestone", "deadline", "kickoff", "standup",
    "retro", "code review", "merge conflict", "feature branch",
    "hotfix branch", "dependency", "package", "library", "framework",
)

# Tech entries that are grammatically plural: {tech_s} templates exclude these
# so "the API keys is down" can never occur.
PLURAL_TECH = ("API keys", "SSH keys", "log files", "release notes")
TECH_SINGULAR = tuple(t for t in TECH_TERMS if t not in PLURAL_TECH)

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
)
WORD_NUMBERS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
)

WEEKDAYS_LOWER = tuple(w.lower() for w in WEEKDAYS)
MONTHS_LOWER = tuple(m.lower() for m in MONTHS)


# --------------------------------------------------------------------------
# Compiled matchers (built once at import)
# --------------------------------------------------------------------------

def _alternation(words, flags=0):
    escaped = sorted({re.escape(w) for w in words}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(escaped) + r")(?![A-Za-z])", flags)


# Names are matched case-sensitively (Title case): clean outputs are well-formed
# sentences, and this keeps common words like "will"/"mark" from matching.
_NAME_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    + "|".join(sorted((re.escape(n) for n in FIRST_NAMES), key=len, reverse=True))
    + r")(?: (?:"
    + "|".join(sorted((re.escape(n) for n in LAST_NAMES), key=len, reverse=True))
    + r"))?(?![A-Za-z])"
)
_ITEM_RE = _alternation(ITEMS, re.IGNORECASE)
_LOCATION_RE = _alternation(LOCATIONS, re.IGNORECASE)
_TECH_RE = _alternation(TECH_TERMS, re.IGNORECASE)

_WEEKDAY_PART = "|".join(w.lower() for w in WEEKDAYS)

_TIME_RES = [
    re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?:\s*(?:am|pm))?", re.IGNORECASE),
    re.compile(r"(?<!\d)\d{1,2}\s*(?:am|pm)", re.IGNORECASE),
    re.compile(r"(?:half past|quarter past|quarter to)\s+\d{1,2}", re.IGNORECASE),
    re.compile(r"(?<!\d)\d{1,2}\s+o'clock", re.IGNORECASE),
    re.compile(r"(?:noon|midnight)", re.IGNORECASE),
]

_DATE_RES = [
    re.compile(r"(?:" + _WEEKDAY_PART + r")(?: (?:morning|afternoon|evening|night))?", re.IGNORECASE),
    re.compile(r"(?:next|this) (?:" + _WEEKDAY_PART + r")", re.IGNORECASE),
    re.compile(
        r"(?:"
        + "|".join(m.lower() for m in MONTHS)
        + r") \d{1,2}(?:st|nd|rd|th)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\d{1,2}(?:st|nd|rd|th) of (?:"
        + "|".join(m.lower() for m in MONTHS)
        + r")",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:today|tomorrow|yesterday|tonight|this week|next week|this weekend|next weekend)",
        re.IGNORECASE,
    ),
]

_NUMBER_RES = [
    re.compile(r"\$\d+(?:\.\d+)?"),
    re.compile(r"\d+(?:\.\d+)+"),  # versions like 2.4.1
    re.compile(r"\d+ percent", re.IGNORECASE),
    re.compile(r"\d+"),
    # Word numbers: also exclude hyphen compounds so "one-on-one" is not matched.
    re.compile(
        r"(?<![A-Za-z-])(?:"
        + "|".join(sorted((re.escape(w) for w in WORD_NUMBERS), key=len, reverse=True))
        + r")(?![A-Za-z-])",
        re.IGNORECASE,
    ),
]

# Matcher order defines extraction priority when spans collide at the same start.
_MATCHERS = (
    ("time", _TIME_RES),
    ("date", _DATE_RES),
    ("name", (_NAME_RE,)),
    ("item", (_ITEM_RE,)),
    ("location", (_LOCATION_RE,)),
    ("tech", (_TECH_RE,)),
    ("number", _NUMBER_RES),
)


# --------------------------------------------------------------------------
# Span extraction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    kind: str
    canonical: str
    raw: str


def entity_spans(text: str) -> list[EntitySpan]:
    """Find all entity spans, resolving overlaps by earliest start, longest span."""
    candidates: list[tuple[int, int, str, str]] = []  # (start, end, kind, raw)
    for kind, res in _MATCHERS:
        for regex in res:
            for m in regex.finditer(text):
                candidates.append((m.start(), m.end(), kind, m.group(0)))
    # Longest span first (ties: earliest start). This resolves cross-kind
    # embedding, e.g. "the beach towel": the item span "beach towel" must win
    # over the location span "the beach", and "Room 302" (location) must win
    # over the number "302". Overlap is checked against ALL kept intervals —
    # a simple last_end cursor would wrongly reject non-overlapping spans that
    # start before the current cursor.
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    kept: list[EntitySpan] = []
    kept_intervals: list[tuple[int, int]] = []
    for start, end, kind, raw in candidates:
        if any(start < k_end and end > k_start for k_start, k_end in kept_intervals):
            continue  # overlaps an already-kept (longer) span
        kept.append(EntitySpan(start=start, end=end, kind=kind, canonical=normalize(raw), raw=raw))
        kept_intervals.append((start, end))
    kept.sort(key=lambda s: s.start)
    return kept


def extract_entities(text: str) -> set[str]:
    return {span.canonical for span in entity_spans(text)}


def pick_entity_span(text: str, rng: random.Random, kinds: tuple[str, ...] | str) -> EntitySpan:
    kinds = (kinds,) if isinstance(kinds, str) else kinds
    spans = [s for s in entity_spans(text) if s.kind in kinds]
    if not spans:
        raise SlotError(f"no entity of kind {kinds!r} in {text!r}")
    return rng.choice(spans)


# --------------------------------------------------------------------------
# Value makers (kept values for clean outputs)
# --------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{('th', 'st', 'nd', 'rd', 'th')[min(n % 10, 4)]}"


def make_number(rng: random.Random, family: str = "number") -> str:
    """Generic {number} slots yield small conversational values ONLY (1-999 or
    word numbers) — port/id-style 4-5 digit values live in make_port/make_id
    and duration phrases in make_duration, so nonsense like
    "running 60944 minutes late" cannot occur. Price/percent/version families
    are only used where a template slot explicitly asks for them."""
    if family == "price":
        return f"${rng.randint(1, 99)}.{rng.randint(0, 99):02d}"
    if family == "percent":
        return f"{rng.randint(5, 95)} percent"
    if family == "version":
        return f"{rng.randint(1, 9)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
    fam = rng.random()
    if fam < 0.65:
        return str(rng.randint(1, 999))
    return rng.choice(WORD_NUMBERS)


def make_duration(rng: random.Random) -> str:
    """Realistic conversational delay/duration intervals (always digit-form so
    the value stays extractable as a number)."""
    return f"{rng.choice((5, 10, 15, 20, 30, 45, 60, 90))} minutes"


def make_port(rng: random.Random) -> str:
    """Port values reserved for technical templates: common dev ports, plus
    occasional random 4-5 digit values inside the valid TCP range."""
    common = (8080, 3000, 9090, 8000, 5000, 4000, 8443, 9091, 5432, 3306,
              6379, 8081, 3001, 5001, 8082, 9000)
    if rng.random() < 0.8:
        return str(rng.choice(common))
    return str(rng.randint(1024, 65535))


def make_id(rng: random.Random) -> str:
    """4-5 digit reference ids for codes/numbers slots (confirmation codes,
    gate codes, build numbers)."""
    if rng.random() < 0.7:
        return str(rng.randint(1000, 9999))
    return str(rng.randint(10000, 99999))


def make_time(rng: random.Random) -> str:
    fam = rng.random()
    if fam < 0.25:
        return f"{rng.randint(1, 12)} {rng.choice(['AM', 'PM'])}"
    if fam < 0.50:
        return f"{rng.randint(1, 12)}:{rng.choice(['00', '15', '30', '45'])} {rng.choice(['AM', 'PM'])}"
    if fam < 0.70:
        return f"half past {rng.randint(1, 11)}"
    if fam < 0.85:
        return f"quarter {'past' if rng.random() < 0.5 else 'to'} {rng.randint(1, 12)}"
    if fam < 0.95:
        return f"{rng.randint(1, 12)} o'clock"
    return rng.choice(["noon", "midnight"])


def make_date(rng: random.Random) -> str:
    # Note: no bare "today/tomorrow/yesterday" family — those read wrong after
    # "on {date}" ("on tomorrow"). Relative days still appear in training as
    # ABANDONED values via _date_variant, which is where they are natural.
    fam = rng.random()
    if fam < 0.35:
        return rng.choice(WEEKDAYS)
    if fam < 0.50:
        return f"next {rng.choice(WEEKDAYS)}"
    if fam < 0.75:
        month = rng.choice(MONTHS)
        day = rng.randint(1, 28)
        return f"{month} {_ordinal(day) if rng.random() < 0.7 else day}"
    if fam < 0.90:
        wd = rng.choice(WEEKDAYS)
        part = rng.choice(["morning", "afternoon", "evening", "night"])
        return f"{wd} {part}"
    # No self-articled "the Nth of Month" here: it would double up with
    # templates that already write "the {date}" ("the the 18th of August").
    # Extraction still supports that form for stress-test data and variants.
    month = rng.choice(MONTHS)
    day = rng.randint(1, 28)
    return f"{month} {_ordinal(day) if rng.random() < 0.5 else day}"


def make_name(rng: random.Random) -> str:
    first = rng.choice(FIRST_NAMES)
    if rng.random() < 0.3:
        return f"{first} {rng.choice(LAST_NAMES)}"
    return first


# --------------------------------------------------------------------------
# Same-kind variants (abandoned values for corrections)
# --------------------------------------------------------------------------

def entity_variant(span: EntitySpan, rng: random.Random) -> str:
    raw = span.raw
    kind = span.kind
    if kind == "name":
        return _name_variant(raw, rng)
    if kind == "item":
        return _pool_variant(raw, ITEMS, rng)
    if kind == "location":
        return _pool_variant(raw, LOCATIONS, rng)
    if kind == "tech":
        return _pool_variant(raw, TECH_TERMS, rng)
    if kind == "number":
        return _number_variant(raw, rng)
    if kind == "time":
        return _time_variant(raw, rng)
    if kind == "date":
        return _date_variant(raw, rng)
    raise ValueError(f"unknown entity kind {kind!r}")


def _pool_variant(raw: str, pool, rng: random.Random) -> str:
    choices = [v for v in pool if v.lower() != raw.lower()]
    return rng.choice(choices)


def _name_variant(raw: str, rng: random.Random) -> str:
    parts = raw.split()
    first, rest = parts[0], parts[1:]
    new_first = rng.choice([n for n in FIRST_NAMES if n.lower() != first.lower()])
    return new_first + ((" " + " ".join(rest)) if rest else "")


def _number_variant(raw: str, rng: random.Random) -> str:
    low = raw.lower()
    if low.startswith("$"):
        while True:
            v = f"${rng.randint(1, 99)}.{rng.randint(0, 99):02d}"
            if v != raw:
                return v
    if low.endswith(" percent"):
        while True:
            v = f"{rng.randint(5, 95)} percent"
            if v.lower() != low:
                return v
    if "." in raw and re.fullmatch(r"(\d+)(?:\.(\d+))+", raw):
        parts = raw.split(".")
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        new_last = str((int(parts[-1]) + delta) % 10)
        v = ".".join(parts[:-1] + [new_last])
        return v if v != raw else ".".join(parts[:-1] + [str((int(parts[-1]) + 1) % 10)])
    if re.fullmatch(r"\d+", raw):
        n = int(raw)
        if n >= 1000:  # port/id-style values stay large (ids reach 99999)
            while True:
                v = n + rng.choice([-900, -500, -100, 100, 500, 900])
                if v != n and 1000 <= v <= 99999:
                    return str(v)
        while True:
            v = n + rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            if v != n and 1 <= v <= 999:
                return str(v)
    while True:  # word-form numbers
        v = rng.choice(WORD_NUMBERS)
        if v != low:
            return v


def _time_variant(raw: str, rng: random.Random) -> str:
    low = raw.lower()
    if low == "noon":
        return "midnight"
    if low == "midnight":
        return "noon"
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?:\s*(am|pm))?", low)
    if m:
        hour, minute, mer = int(m.group(1)), m.group(2), m.group(3)
        while True:
            new_hour = rng.randint(1, 12)
            new_minute = rng.choice(["00", "15", "30", "45"])
            if new_hour == hour and new_minute == minute:
                continue
            return f"{new_hour}:{new_minute}{' ' + mer.upper() if mer else ''}"
    m = re.fullmatch(r"(\d{1,2})\s*(am|pm)", low)
    if m:
        while True:
            new_hour = rng.randint(1, 12)
            if new_hour != int(m.group(1)):
                return f"{new_hour} {m.group(2).upper()}"
    m = re.fullmatch(r"(half past|quarter past|quarter to)\s+(\d{1,2})", low)
    if m:
        while True:
            nv = rng.randint(1, 12)
            if str(nv) != m.group(2):
                return f"{m.group(1)} {nv}"
    m = re.fullmatch(r"(\d{1,2})\s+o'clock", low)
    if m:
        while True:
            nv = rng.randint(1, 12)
            if nv != int(m.group(1)):
                return f"{nv} o'clock"
    return make_time(rng)


def _date_variant(raw: str, rng: random.Random) -> str:
    low = raw.lower()
    if low in WEEKDAYS_LOWER:
        # 30% of the time abandon a relative day instead of another weekday, so
        # the model sees "tomorrow no wait Tuesday"-style corrections in training.
        if rng.random() < 0.3:
            return rng.choice(["today", "tomorrow", "yesterday"])
        while True:
            v = rng.choice(WEEKDAYS)
            if v.lower() != low:
                return v
    if low in ("today", "tomorrow", "yesterday"):
        while True:
            v = rng.choice(["today", "tomorrow", "yesterday"])
            if v != low:
                return v
    m = re.fullmatch(r"(next|this)\s+(\w+)", low)
    if m and m.group(2) in WEEKDAYS_LOWER:
        prefix = m.group(1).capitalize()
        while True:
            wd = rng.choice(WEEKDAYS)
            if wd.lower() != m.group(2):
                return f"{prefix} {wd}"
    m = re.fullmatch(r"(\w+)\s+(\d{1,2})(st|nd|rd|th)?", low)
    if m and m.group(1) in MONTHS_LOWER:
        month = m.group(1).capitalize()
        day, suffix = int(m.group(2)), m.group(3)
        if rng.random() < 0.6:  # same month, different day (adjacent-style correction)
            while True:
                new_day = rng.randint(1, 28)
                if new_day != day:
                    return f"{month} {_ordinal(new_day) if suffix else new_day}"
        new_month = rng.choice([mm for mm in MONTHS if mm.lower() != m.group(1)])
        new_day = rng.randint(1, 28)
        return f"{new_month} {_ordinal(new_day) if suffix else new_day}"
    m = re.fullmatch(r"(\w+)\s+(morning|afternoon|evening|night)", low)
    if m and m.group(1) in WEEKDAYS_LOWER:
        part = m.group(2).capitalize()
        while True:
            wd = rng.choice(WEEKDAYS)
            if wd.lower() != m.group(1):
                return f"{wd} {part}"
    m = re.fullmatch(r"the\s+(\d{1,2})(?:st|nd|rd|th)\s+of\s+(\w+)", low)
    if m and m.group(2) in MONTHS_LOWER:
        month = m.group(2).capitalize()
        while True:
            new_day = rng.randint(1, 28)
            if new_day != int(m.group(1)):
                return f"the {_ordinal(new_day)} of {month}"
    while True:  # covers relative phrases like "this week" / "next weekend"
        v = make_date(rng)
        if v.lower() != low:
            return v


# --------------------------------------------------------------------------
# Without-replacement sampling across enumerated pools
# --------------------------------------------------------------------------

class PoolSampler:
    """Samples pool values without replacement within one record session.

    Prevents one record from ever containing the same name/item/location/tech
    twice, while keeping global stats uniform. Number-restricted kinds
    (item_s / item_p / tech_s) draw from sub-pools and share their used-set
    with the parent pool, so a record never repeats an item across variants.
    """

    def __init__(self, rng: random.Random):
        self._rng = rng
        self.pools = {
            "name": FIRST_NAMES,
            "surname": LAST_NAMES,
            "item": ITEMS,
            "item_s": SINGULAR_ITEMS,
            "item_p": PLURAL_ITEMS,
            "location": LOCATIONS,
            "tech": TECH_TERMS,
            "tech_s": TECH_SINGULAR,
            "tech_p": PLURAL_TECH,
        }
        self._used = {kind: set() for kind in self.pools}
        # Sampling one family member excludes the others from this session.
        self._families = {
            "item": {"item", "item_s", "item_p"},
            "item_s": {"item", "item_s", "item_p"},
            "item_p": {"item", "item_s", "item_p"},
            "tech": {"tech", "tech_s", "tech_p"},
            "tech_s": {"tech", "tech_s", "tech_p"},
            "tech_p": {"tech", "tech_s", "tech_p"},
        }

    def sample(self, kind: str) -> str:
        pool = self.pools[kind]  # KeyError for unknown kinds
        family = self._families.get(kind, {kind})
        available = [
            v for v in pool
            if all(v.lower() not in self._used[k] for k in family)
        ]
        if not available:
            raise SlotError(f"pool {kind} exhausted")
        value = self._rng.choice(available)
        for k in family:
            self._used[k].add(value.lower())
        return value
