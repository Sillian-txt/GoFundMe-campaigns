# GoFundMe Campaign Scraper + Sentiment Analysis
# Requirements: pip install selenium webdriver-manager beautifulsoup4 textblob
# After install, run once: python -m textblob.download_corpora

# ── 1. LIBRARIES ─────────────────────────────────────────────────────────────

import re
import csv
import time
import random
import logging

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from textblob import TextBlob

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── USER CONTROLS ─────────────────────────────────────────────────────────────

CAMPAIGNS_PER_CATEGORY = 1          # campaigns to scrape per category
OUTPUT_FILE            = "gofundme_campaigns.csv"
HEADLESS               = True       # set False to watch the browser


# ── 2. CATEGORIES ─────────────────────────────────


CATEGORIES = {
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

# ── 3. SELENIUM SETUP ────────────────────────────────────────────────────────

def build_driver() -> webdriver.Chrome:
    options = Options()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver

# ── SMART WAIT ────────────────────────────────────────────────────────────────
# Two modes:
#   "discover" — after loading a category index page (heavier, less frequent)
#   "campaign" — between individual campaign pages (lighter, more frequent)
#
# Every ~8 requests a longer "human pause" is injected to break rhythm.
# All intervals are jittered so the timing pattern is never regular.

_request_count = 0

def smart_wait(mode: str = "campaign") -> None:
    global _request_count
    _request_count += 1

    base = random.uniform(4.0, 7.0) if mode == "discover" else random.uniform(2.0, 4.5)

    # Occasional longer human-like pause every ~8 requests
    if _request_count % random.randint(7, 10) == 0:
        base += random.uniform(6.0, 12.0)
        log.info(f"  [human pause] total wait ≈ {base:.1f}s")
    else:
        log.info(f"  [wait/{mode}] {base:.1f}s")

    time.sleep(base)

# ── 4. DISCOVER: COLLECT CAMPAIGN URLs ───────────────────────────────────────

def get_campaign_urls(driver: webdriver.Chrome, category: str, url: str, limit: int) -> list[str]:
    log.info(f"[discover] {category} → {url}")
    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )
    except Exception:
        log.warning(f"  timed out loading discover page for '{category}'")
        return []

    smart_wait("discover")
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Campaign links follow the pattern /f/<slug>
    pattern = re.compile(r"^https://www\.gofundme\.com/f/[^/?#]+")
    seen, urls = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("http"):
            href = "https://www.gofundme.com" + href
        if pattern.match(href) and href not in seen:
                href = href.split("?")[0]  # strip ?qid=... tracking parameter
                seen.add(href)
                urls.append(href)
                if len(urls) >= limit:
                    break

    log.info(f"  found {len(urls)} campaign URLs")
    return urls

# ── 5. SCRAPE RAW DATA FROM A CAMPAIGN PAGE ──────────────────────────────────

def _parse_goal(raw: str) -> str | None:
    """
    Convert GoFundMe's formatted goal string to a plain integer string.
    Handles shorthand like '£1.6K' → '1600', '$2.5M' → '2500000',
    and formatted amounts like '£1,171' → '1171'.
    """
    if not raw:
        return None
    cleaned = re.sub(r"[£$€\s,]", "", raw).upper()
    for suffix, mult in [("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)]:
        if cleaned.endswith(suffix):
            try:
                return str(int(float(cleaned[:-1]) * mult))
            except ValueError:
                return None
    numeric = re.sub(r"[^\d.]", "", cleaned)
    return str(int(float(numeric))) if numeric else None


def scrape_campaign(driver: webdriver.Chrome, url: str, category: str) -> dict:
    """
    Load a campaign page and return a dict with:
      url, category, goal_amount, donor_count, main_text (internal, not exported).
    """
    record = {"url": url, "category": category,
              "goal_amount": None, "donor_count": None, "main_text": None}

    driver.get(url)
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
    except Exception:
        log.warning(f"  timed out: {url}")
        return record

    # Expand full story — button is labelled "read more" on GFM
    try:
        btn = driver.find_element(
            By.XPATH,
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
            " 'abcdefghijklmnopqrstuvwxyz'), 'read more')]",
        )
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(1)
    except Exception:
        pass

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # ── Goal amount ───────────────────────────────────────────────────────────
    # location: data-testid="static-donation-overview"
    # inside a <button data-tracking-id="fundraiser goal clicked">
    overview = soup.find(attrs={"data-testid": "static-donation-overview"})
    if overview:
        goal_btn = overview.find(attrs={"data-tracking-id": "fundraiser goal clicked"})
        if goal_btn:
            record["goal_amount"] = _parse_goal(goal_btn.get_text(strip=True))

    # ── Donor count ───────────────────────────────────────────────────────────
    # location: data-testid="hero-story-slide-content"
    # as a <span> with text like "45 donors"
    hero = soup.find(attrs={"data-testid": "hero-story-slide-content"})
    if hero:
        donor_span = hero.find("span", string=re.compile(r"\d+\s+donors?", re.I))
        if donor_span:
            match = re.search(r"\d+", donor_span.get_text())
            if match:
                record["donor_count"] = match.group()

    # ── Main text (used for NLP only — not written to CSV) ───────────────────
    # location: data-testid="campaign-description"
    # as plain <div> blocks (no <p> tags)
    desc = soup.find(attrs={"data-testid": "campaign-description"})
    if desc:
        record["main_text"] = desc.get_text(separator=" ", strip=True)

    return record

# ── 6 & 7. TEXTBLOB PREPROCESSING + SENTIMENT ANALYSIS ──────────────────────

GAIN_WORDS = {
    "achieve", "benefit", "build", "chance", "create", "dream", "earn",
    "empower", "flourish", "gain", "grow", "help", "hope", "improve",
    "invest", "joy", "opportunity", "positive", "progress", "prosper",
    "reach", "recover", "restore", "reward", "rise", "save", "succeed",
    "success", "support", "thrive", "together", "transform", "win", "fund",
    "enable", "future", "bright", "goal", "possible", "better", "grateful",
    "thankful", "blessed", "give", "donate",
}

LOSS_WORDS = {
    "afraid", "anguish", "battle", "broke", "burden", "cancer", "cannot",
    "crisis", "danger", "death", "debt", "desperate", "destroy", "devastate",
    "difficult", "disease", "dying", "emergency", "fail", "fear", "fight",
    "hard", "heartbreak", "homeless", "hopeless", "hurt", "illness", "injury",
    "loss", "lose", "lost", "need", "nightmare", "pain", "panic", "problem",
    "risk", "ruin", "scared", "serious", "sick", "struggle", "suffer",
    "tragedy", "trauma", "unable", "urgent", "victim", "vulnerable", "worry",
    "dire", "critical", "overwhelming", "exhausted", "helpless",
}

INCLUSIVE_WORDS = {
    "we", "our", "ours", "ourselves", "us", "together", "community",
    "everyone", "family", "join", "team", "united", "collective", "share",
}

EXCLUSIVE_WORDS = {"i", "me", "my", "mine", "myself", "you", "your", "you're"}

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z']+\b", text.lower())

def _ratio(hits: int, total: int) -> float:
    return hits / total if total else 0.0

def analyze(main_text: str | None) -> dict:
    """
    Preprocess text with TextBlob and return sentiment scores,
    each on a -1 … +1 scale.

      gain_loss_score    +1 = pure gain framing,   -1 = pure loss framing
      emotional_valence  +1 = positive emotion,    -1 = negative  (TextBlob polarity)
      inclusivity_score  +1 = inclusive language,  -1 = exclusive language
    """
    empty = {"gain_loss_score": None, "emotional_valence": None,
             "inclusivity_score": None}

    if not main_text:
        return empty

    blob   = TextBlob(main_text)
    tokens = _tokenize(main_text)
    total  = len(tokens) or 1

    gain_r = _ratio(sum(1 for t in tokens if t in GAIN_WORDS), total)
    loss_r = _ratio(sum(1 for t in tokens if t in LOSS_WORDS), total)
    incl_r = _ratio(sum(1 for t in tokens if t in INCLUSIVE_WORDS), total)
    excl_r = _ratio(sum(1 for t in tokens if t in EXCLUSIVE_WORDS), total)

    return {
        "gain_loss_score":       round(gain_r - loss_r, 4),
        "emotional_valence":     round(blob.sentiment.polarity, 4),
        "inclusivity_score":     round(incl_r - excl_r, 4),
    }

# ── 8. EXPORT TO CSV ─────────────────────────────────────────────────────────

CSV_FIELDS = [
    "url",                    # primary key
    "category",               # GoFundMe category
    "goal_amount",            # monetary goal (numeric string)
    "donor_count",            # number of donors
    "gain_loss_score",        # +1 gain-framed … -1 loss-framed
    "emotional_valence",      # +1 positive … -1 negative
    "inclusivity_score",      # +1 inclusive … -1 exclusive
]


def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved {len(rows)} rows → {path}")

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    categories = list(CATEGORIES.items())
    random.shuffle(categories)
    log.info(f"Running across all {len(categories)} categories (shuffled)")

    driver = build_driver()
    all_rows: list[dict] = []

    try:
        for cat_name, cat_url in categories:
            urls = get_campaign_urls(driver, cat_name, cat_url, limit=CAMPAIGNS_PER_CATEGORY)

            for i, url in enumerate(urls, 1):
                log.info(f"[{cat_name}] {i}/{len(urls)}: {url}")
                record = scrape_campaign(driver, url, cat_name)
                scores = analyze(record.pop("main_text"))   # NLP then discard text
                all_rows.append({**record, **scores})
                log.info(
                    f"  goal={record['goal_amount']}  donors={record['donor_count']}  "
                    f"gain/loss={scores['gain_loss_score']}  "
                    f"valence={scores['emotional_valence']}  "
                    f"inclusive={scores['inclusivity_score']}"
                )
                smart_wait("campaign")

    finally:
        driver.quit()

    save_csv(all_rows, OUTPUT_FILE)


if __name__ == "__main__":
    main()
