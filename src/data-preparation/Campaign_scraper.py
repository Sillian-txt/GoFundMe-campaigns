# GoFundMe Campaign Scraper + Sentiment Analysis
# Requirements:
#   pip install selenium webdriver-manager beautifulsoup4 empath nltk
# After install, run once:
#   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt_tab')"
#
# Usage:
#   python gofundme_scraper.py                        # defaults
#   python gofundme_scraper.py --limit 5 --no-headless --out results.csv

# ── 1. LIBRARIES ──────────────────────────────────────────────────────────────

import re
import csv
import time
import random
import logging
import argparse
import contextlib
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from empath import Empath
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

# Initialise once at module level — both are expensive to construct
_empath = Empath()
_vader  = SentimentIntensityAnalyzer()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── 2. CATEGORIES ─────────────────────────────────────────────────────────────

CATEGORIES: dict[str, str] = {
    "Medical":     "https://www.gofundme.com/discover/medical-fundraiser",
    "Memorial":    "https://www.gofundme.com/discover/memorial-fundraiser",
    "Emergency":   "https://www.gofundme.com/discover/emergency-fundraiser",
    "Charity":     "https://www.gofundme.com/discover/charity-fundraiser",
    "Education":   "https://www.gofundme.com/discover/education-fundraiser",
    "Animal":      "https://www.gofundme.com/discover/animal-fundraiser",
    "Environment": "https://www.gofundme.com/discover/environment-fundraiser",
    "Business":    "https://www.gofundme.com/discover/business-fundraiser",
    "Community":   "https://www.gofundme.com/discover/community-fundraiser",
    "Competition": "https://www.gofundme.com/discover/competition-fundraiser",
    "Creative":    "https://www.gofundme.com/discover/creative-fundraiser",
    "Event":       "https://www.gofundme.com/discover/event-fundraiser",
    "Faith":       "https://www.gofundme.com/discover/faith-fundraiser",
    "Family":      "https://www.gofundme.com/discover/family-fundraiser",
    "Sports":      "https://www.gofundme.com/discover/sports-fundraiser",
    "Travel":      "https://www.gofundme.com/discover/travel-fundraiser",
    "Volunteer":   "https://www.gofundme.com/discover/volunteer-fundraiser",
    "Wishes":      "https://www.gofundme.com/discover/wishes-fundraiser",
}

# ── 3. CLI / CONFIG ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GoFundMe scraper + sentiment analysis")
    p.add_argument("--limit",       type=int,   default=3,
                   help="Campaigns to scrape per category (default: 1)")
    p.add_argument("--out",         type=str,   default="gofundme_campaigns.csv",
                   help="Output CSV path (default: gofundme_campaigns.csv)")
    p.add_argument("--no-headless", action="store_true",
                   help="Show the browser window")
    p.add_argument("--retries",     type=int,   default=2,
                   help="Retry attempts per page on timeout (default: 2)")
    return p.parse_args()

# ── 4. SELENIUM SETUP ─────────────────────────────────────────────────────────

# Cache the driver binary path so ChromeDriverManager only hits the network once
# per interpreter session (important when scraping many categories in one run).
_DRIVER_PATH: Optional[str] = None

def _get_driver_path() -> str:
    global _DRIVER_PATH
    if _DRIVER_PATH is None:
        _DRIVER_PATH = ChromeDriverManager().install()
    return _DRIVER_PATH


def build_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--lang=en-US")
    prefs = {"intl.accept_languages": "en-US,en"}
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(_get_driver_path()),
        options=options,
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver

# ── 5. SMART WAIT ─────────────────────────────────────────────────────────────
# Two modes:
#   "discover" — after loading a category index page (heavier, less frequent)
#   "campaign" — between individual campaign pages (lighter, more frequent)
#
# Every ~8 requests a longer "human pause" is injected to break rhythm.
# All intervals are jittered so the timing pattern is never regular.
# State is encapsulated in a class so there is no mutable global.

class RateLimiter:
    def __init__(self) -> None:
        self._count = 0
        self._next_long_pause_at = random.randint(7, 10)

    def wait(self, mode: str = "campaign") -> None:
        self._count += 1
        base = random.uniform(4.0, 7.0) if mode == "discover" else random.uniform(2.0, 4.5)

        if self._count >= self._next_long_pause_at:
            base += random.uniform(6.0, 12.0)
            self._next_long_pause_at = self._count + random.randint(7, 10)
            log.info(f"  [human pause] total wait ≈ {base:.1f}s")
        else:
            log.info(f"  [wait/{mode}] {base:.1f}s")

        time.sleep(base)

# ── 6. DISCOVER: COLLECT CAMPAIGN URLs ───────────────────────────────────────

_CAMPAIGN_URL_RE = re.compile(r"^https://www\.gofundme\.com/f/[^/?#]+")


def get_campaign_urls(
    driver: webdriver.Chrome,
    category: str,
    url: str,
    limit: int,
    limiter: RateLimiter,
    retries: int = 2,
) -> list[str]:
    log.info(f"[discover] {category} → {url}")

    for attempt in range(1, retries + 2):   # +2 so range covers (retries+1) total tries
        driver.get(url)
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "a"))
            )
            break
        except Exception:
            if attempt <= retries:
                log.warning(f"  timeout on discover page (attempt {attempt}), retrying…")
                time.sleep(random.uniform(3.0, 6.0))
            else:
                log.warning(f"  giving up on discover page for '{category}' after {attempt} attempts")
                return []

    limiter.wait("discover")
    soup = BeautifulSoup(driver.page_source, "html.parser")

    seen: set[str] = set()
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if not href.startswith("http"):
            href = "https://www.gofundme.com" + href
        href = href.split("?")[0]           # strip ?qid=… tracking parameter
        if _CAMPAIGN_URL_RE.match(href) and href not in seen:
            seen.add(href)
            urls.append(href)
            if len(urls) >= limit:
                break

    log.info(f"  found {len(urls)} campaign URLs")
    return urls

# ── 7. SCRAPE RAW DATA FROM A CAMPAIGN PAGE ───────────────────────────────────

_CURRENCY_RE  = re.compile(r"[£$€\s,]")
_DONOR_RE     = re.compile(r"(\d[\d,]*)\s+donors?", re.I)


def _parse_goal(raw: str) -> Optional[str]:
    """
    Convert GoFundMe's formatted goal string to a plain integer string.
    Handles shorthand like '£1.6K' → '1600', '$2.5M' → '2500000',
    and formatted amounts like '£1,171' → '1171'.
    Strips leading non-numeric text (e.g. "goal: £1K") before parsing.
    """
    if not raw:
        return None
    # Pre-strip: drop everything before the first currency symbol or digit
    raw = re.sub(r"^[^\d£$€]*", "", raw)
    cleaned = _CURRENCY_RE.sub("", raw).upper().strip()
    if not cleaned:
        return None
    for suffix, mult in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if cleaned.endswith(suffix):
            with contextlib.suppress(ValueError):
                return str(int(float(cleaned[:-1]) * mult))
            return None
    numeric = re.sub(r"[^\d.]", "", cleaned)
    if not numeric:
        return None
    with contextlib.suppress(ValueError):
        return str(int(float(numeric)))
    return None


def _parse_donors(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract donor count with two fallback strategies:
      1. Preferred: data-testid="hero-story-slide-content" <span> with "N donors"
      2. Fallback:  any element on the page whose text matches the donor pattern
    """
    # Strategy 1 — preferred location
    hero = soup.find(attrs={"data-testid": "hero-story-slide-content"})
    if hero:
        span = hero.find("span", string=_DONOR_RE)
        if span:
            m = _DONOR_RE.search(span.get_text())
            if m:
                return m.group(1).replace(",", "")

    # Strategy 2 — page-wide fallback
    for tag in soup.find_all(string=_DONOR_RE):
        m = _DONOR_RE.search(tag)
        if m:
            return m.group(1).replace(",", "")

    return None


# Matches currency-prefixed amounts anywhere in a string, e.g. "$10,000 goal"
_GOAL_INLINE_RE = re.compile(
    r"[£$€]\s*[\d,]+(?:\.\d+)?\s*(?:[KkMmBb](?:illion)?)?", re.I
)


def _parse_goal_from_page(soup: BeautifulSoup, driver: webdriver.Chrome) -> Optional[str]:
    """
    Six-stage fallback chain for extracting the fundraising goal.
    Returns the first non-None result, or None if all stages fail.

    Stage 1 — data-testid="static-donation-overview": parse overview text
              directly for the "raised of £X" pattern GoFundMe now uses.
              The old data-tracking-id="fundraiser goal clicked" button has
              been removed from GoFundMe's markup (confirmed by diagnostic).
    Stage 2 — any element whose aria-label contains "goal"
    Stage 3 — page-wide search for an element with class containing "goal"
    Stage 4 — JSON-LD structured data (<script type="application/ld+json">)
    Stage 5 — regex scan of full page text for currency near "goal" OR near
              the "raised of £X" / "of £X" construction GoFundMe actually uses
    Stage 6 — extract the second distinct currency amount from the overview
              widget text (raised amount is first, goal is second)
    """
    # Stage 1: parse the overview container text for "raised of £X"
    # Diagnostic confirmed: the container IS present but the internal button
    # data-tracking-id has been removed. The goal appears as the currency
    # amount following "of" (with optional non-breaking space) in the text:
    #   e.g. "11 % £42,897 raised of £410K"
    _RAISED_OF_RE = re.compile(
        r"of\s*[\xa0\s]*([£$€]\s*[\d,]+(?:\.\d+)?\s*[KkMmBb]?)", re.I
    )
    overview = soup.find(attrs={"data-testid": "static-donation-overview"})
    if overview:
        overview_text = overview.get_text(separator=" ")
        m = _RAISED_OF_RE.search(overview_text)
        if m:
            result = _parse_goal(m.group(1))
            if result:
                log.debug("  goal via stage 1 (raised-of pattern)")
                return result

    # Stage 2: aria-label containing "goal"
    for tag in soup.find_all(attrs={"aria-label": re.compile(r"goal", re.I)}):
        result = _parse_goal(tag.get_text(strip=True))
        if result:
            log.debug("  goal via stage 2 (aria-label)")
            return result

    # Stage 3: any element whose class string contains "goal"
    for tag in soup.find_all(class_=re.compile(r"goal", re.I)):
        text = tag.get_text(strip=True)
        if re.search(r"[\d,]+", text):
            result = _parse_goal(text)
            if result:
                log.debug("  goal via stage 3 (class)")
                return result

    # Stage 4: JSON-LD structured data
    import json
    for script in soup.find_all("script", type="application/ld+json"):
        with contextlib.suppress(Exception):
            data = json.loads(script.string or "")
            for key in ("fundingGoal", "price", "amount"):
                val = data.get(key)
                if val:
                    result = _parse_goal(str(val))
                    if result:
                        log.debug(f"  goal via stage 4 (JSON-LD key={key})")
                        return result

    # Stage 5: regex scan — currency amount near "goal" OR near "of" in a
    # "raised of £X" construction.  Diagnostic showed GoFundMe uses "of £410K"
    # not "goal £410K", so both keywords must be checked.
    page_text = soup.get_text(separator=" ")
    for m in _GOAL_INLINE_RE.finditer(page_text):
        surrounding = page_text[max(0, m.start() - 60): m.end() + 60]
        if re.search(r"\bgoal\b|\braised\b", surrounding, re.I):
            result = _parse_goal(m.group())
            if result:
                log.debug("  goal via stage 5 (regex scan)")
                return result

    # Stage 6: from the overview widget, collect all distinct currency amounts
    # in order — GoFundMe renders them as "£raised of £goal", so the goal is
    # the second distinct amount found.
    if overview:
        amounts = _GOAL_INLINE_RE.findall(
            overview.get_text(separator=" ")
        )
        # Deduplicate while preserving order
        seen_amounts: list[str] = []
        for amt in amounts:
            normalised = re.sub(r"\s+", "", amt)
            if normalised not in seen_amounts:
                seen_amounts.append(normalised)
        if len(seen_amounts) >= 2:
            result = _parse_goal(seen_amounts[1])
            if result:
                log.debug("  goal via stage 6 (second amount in overview)")
                return result

    log.debug("  goal: all stages failed")
    return None


def scrape_campaign(
    driver: webdriver.Chrome,
    url: str,
    category: str,
    campaign_id: str,
    retries: int = 2,
) -> dict:
    """
    Load a campaign page and return a dict with:
      campaign_id, category, goal_amount, donor_count, main_text (internal — not exported).
    campaign_id is an externally assigned sequential identifier; the URL is
    used only for navigation and is not stored.
    Retries up to `retries` additional times on timeout.
    """
    blank: dict = {
        "campaign_id": campaign_id, "category": category,
        "goal_amount": None, "donor_count": None, "main_text": None,
    }

    for attempt in range(1, retries + 2):
        driver.get(url)
        try:
            # Wait for h1 — confirms the page shell has loaded
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "h1"))
            )

            # GoFundMe's donation widget is below the fold and uses
            # intersection-observer lazy loading: the async data fetch that
            # populates goal amount and donor count is only triggered when
            # the widget scrolls into the viewport.  In a headless browser
            # nothing scrolls automatically, so we must do it explicitly
            # before issuing the element wait.
            driver.execute_script("window.scrollTo(0, 600);")
            time.sleep(0.5)   # brief pause for the observer to fire

            # Wait for any known donation-widget selector to appear.
            # Multiple selector variants handle GoFundMe's A/B layout changes.
            _WIDGET_SELECTORS = (
                "[data-testid='static-donation-overview']",
                "[data-testid='donation-overview']",
                "[data-testid='hero-story-slide-content']",
                "[data-testid='fund-raised']",
            )
            for sel in _WIDGET_SELECTORS:
                with contextlib.suppress(Exception):
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                    )
                    break   # stop as soon as one selector resolves

            break   # page loaded successfully — exit retry loop

        except Exception:
            if attempt <= retries:
                log.warning(f"  timeout (attempt {attempt}): {campaign_id}, retrying…")
                time.sleep(random.uniform(3.0, 6.0))
            else:
                log.warning(f"  giving up after {attempt} attempts: {campaign_id}")
                return blank

    # Expand full story — button is labelled "read more" on GFM
    with contextlib.suppress(Exception):
        btn = driver.find_element(
            By.XPATH,
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
            " 'abcdefghijklmnopqrstuvwxyz'), 'read more')]",
        )
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(1)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # ── Goal amount — 5-stage fallback chain ─────────────────────────────────
    goal_amount: Optional[str] = _parse_goal_from_page(soup, driver)

    # ── Donor count ───────────────────────────────────────────────────────────
    donor_count = _parse_donors(soup)

    # ── Main text (used for NLP only — not written to CSV) ───────────────────
    main_text: Optional[str] = None
    desc = soup.find(attrs={"data-testid": "campaign-description"})
    if desc:
        main_text = desc.get_text(separator=" ", strip=True)

    return {
        "campaign_id": campaign_id, "category": category,
        "goal_amount": goal_amount,
        "donor_count": donor_count,
        "main_text": main_text,
    }

# ── 8. SENTIMENT / FRAMING ANALYSIS ──────────────────────────────────────────
#
# gain_loss_score   — Empath lexicon categories (Fast et al. 2016), ratio-scaled.
# emotional_valence — VADER compound score (Hutto & Gilbert 2014).
# inclusivity_score — Direct pronoun token counting, ratio-scaled.
#
# SCORE RANGE NOTE
# ────────────────
# With Empath's normalize=True each category returns a small proportion of
# tokens (typically 0.002–0.05). A raw difference (gain − loss) therefore
# sits in practice around [−0.15, +0.15], making a stated "−1…+1 scale"
# misleading — the extremes are never reached.
#
# Instead all bipolar scores use a RATIO formulation:
#   score = (A − B) / (A + B + ε)
# This equals +1 when only A-words are present, −1 when only B-words are
# present, and 0 when A = B or both are absent. The full range is genuinely
# reachable and the scale is interpretable. (Turney & Littman, 2003)

_GAIN_CATS: list[str] = ["achievement", "positive_emotion", "gain", "optimism",
                          "help", "health", "giving"]
_LOSS_CATS: list[str] = ["suffering", "negative_emotion", "risk", "pain",
                          "death", "anger", "sadness", "fear"]

# Pronouns are a grammatically closed class — direct token counting is more
# reliable than any distributional lexicon, and avoids the Empath category
# naming problem ("we" and "i" are not valid built-in Empath category names).
_INCL_PRONOUNS: frozenset[str] = frozenset({"we", "our", "ours", "ourselves", "us"})
_EXCL_PRONOUNS: frozenset[str] = frozenset({"i", "me", "my", "mine", "myself", "you", "your", "yourself", "you're"})

_TOKEN_RE = re.compile(r"\b[a-z']+\b")
_RATIO_EPS = 1e-9   # prevents division by zero when both sides are 0


def _empath_score(text: str, categories: list[str]) -> float:
    """Sum of Empath normalised category scores across `categories`."""
    scores = _empath.analyze(text, categories=categories, normalize=True)
    if not scores:
        return 0.0
    return sum(v for v in scores.values() if v is not None)


def _ratio_score(a: float, b: float) -> float:
    """
    Bipolar ratio score on [−1, +1]:
      +1  when a > 0 and b = 0
      −1  when b > 0 and a = 0
       0  when a = b (including both zero)
    """
    return (a - b) / (a + b + _RATIO_EPS)


def analyze(main_text: Optional[str]) -> dict:
    """
    Return NLP scores, each on a genuine −1 … +1 ratio scale:
      gain_loss_score    +1 = pure gain framing,   −1 = pure loss framing
      emotional_valence  +1 = positive affect,     −1 = negative (VADER compound)
      inclusivity_score  +1 = fully inclusive,     −1 = fully exclusive
    """
    empty = {"gain_loss_score": None, "emotional_valence": None, "inclusivity_score": None}
    if not main_text or not main_text.strip():
        return empty

    # ── Gain / loss framing via Empath ratio ─────────────────────────────────
    gain = _empath_score(main_text, _GAIN_CATS)
    loss = _empath_score(main_text, _LOSS_CATS)
    gain_loss = _ratio_score(gain, loss)

    # ── Emotional valence via VADER — sentence-level averaging ───────────────
    sentences = sent_tokenize(main_text)
    if sentences:
        valence = sum(
            _vader.polarity_scores(s)["compound"] for s in sentences
        ) / len(sentences)
    else:
        valence = _vader.polarity_scores(main_text)["compound"]

    # ── Inclusivity via direct pronoun token counting ─────────────────────────
    tokens = _TOKEN_RE.findall(main_text.lower())
    incl = sum(1 for t in tokens if t in _INCL_PRONOUNS)
    excl = sum(1 for t in tokens if t in _EXCL_PRONOUNS)
    inclusivity = _ratio_score(incl, excl)

    return {
        "gain_loss_score":   round(gain_loss,   4),
        "emotional_valence": round(valence,      4),
        "inclusivity_score": round(inclusivity,  4),
    }

# ── 9. INCREMENTAL CSV WRITER ─────────────────────────────────────────────────

CSV_FIELDS = [
    "campaign_id",          # primary key — sequential integer string e.g. "0001"
    "category",             # GoFundMe category
    "goal_amount",          # monetary goal (numeric string)
    "donor_count",          # number of donors
    "gain_loss_score",      # +1 gain-framed … -1 loss-framed
    "emotional_valence",    # +1 positive … -1 negative
    "inclusivity_score",    # +1 inclusive … -1 exclusive
]

class CsvWriter:
    """
    Opens the CSV once and flushes each row immediately so that a mid-run
    crash does not lose already-scraped data.
    """
    def __init__(self, path: str) -> None:
        self._path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )
        self._writer.writeheader()
        self._count = 0

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()
        self._count += 1

    def close(self) -> None:
        self._file.close()
        log.info(f"Saved {self._count} rows → {self._path}")

    def __enter__(self) -> "CsvWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    headless = not args.no_headless

    categories = list(CATEGORIES.items())
    random.shuffle(categories)
    log.info(f"Running across all {len(categories)} categories (shuffled), "
             f"limit={args.limit}, headless={headless}, retries={args.retries}")

    limiter  = RateLimiter()
    driver   = build_driver(headless=headless)
    seq      = 0   # sequential campaign counter — zero-padded in the CSV

    with CsvWriter(args.out) as writer:
        try:
            for cat_name, cat_url in categories:
                urls = get_campaign_urls(
                    driver, cat_name, cat_url,
                    limit=args.limit,
                    limiter=limiter,
                    retries=args.retries,
                )

                for i, url in enumerate(urls, 1):
                    seq += 1
                    campaign_id = f"{seq:04d}"  
                    log.info(f"[{cat_name}] {i}/{len(urls)}: {campaign_id}")

                    record = scrape_campaign(
                        driver, url, cat_name,
                        campaign_id=campaign_id,
                        retries=args.retries,
                    )
                    main_text = record.pop("main_text")  
                    scores    = analyze(main_text)
                    row       = {**record, **scores}

                    writer.write(row)
                    log.info(
                        f"  goal={row['goal_amount']}  donors={row['donor_count']}  "
                        f"gain/loss={scores['gain_loss_score']}  "
                        f"valence={scores['emotional_valence']}  "
                        f"inclusive={scores['inclusivity_score']}"
                    )
                    limiter.wait("campaign")

        finally:
            driver.quit()

if __name__ == "__main__":
    main()
