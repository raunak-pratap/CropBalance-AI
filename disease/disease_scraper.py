"""
disease_scraper.py
--------------------------
Scrapes publicly available crop disease advisories from:
  1. ICAR (Indian Council of Agricultural Research)  — icar.org.in
  2. Agri Portal India                               — agricoop.nic.in
  3. PlantVillage (Penn State)                       — plantvillage.psu.edu
  4. Vikaspedia (MeitY)                              — vikaspedia.in

Scraped data is used to:
  - Enrich DISEASE_META with up-to-date pesticide recommendations
  - Build a training FAQ corpus for the advisory chatbot
  - Keep treatment advice current with government guidelines

Usage:
    scraper = DiseaseAdvisoryScraper()
    advisories = scraper.scrape_all()          # returns list of Advisory dicts
    scraper.save(advisories, "data/raw/disease_advisories.json")
"""

import time
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from loguru import logger
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


# ──────────────────────────────────────────────
# Data schema
# ──────────────────────────────────────────────

@dataclass
class Advisory:
    source:       str
    url:          str
    crop:         str
    disease:      str
    symptoms:     str
    treatment:    str
    scraped_at:   str
    hash:         str = ""          # deduplication fingerprint

    def __post_init__(self):
        raw = f"{self.source}{self.crop}{self.disease}{self.treatment}"
        self.hash = hashlib.md5(raw.encode()).hexdigest()


# ──────────────────────────────────────────────
# Base scraper
# ──────────────────────────────────────────────

class BaseScraper:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SmartFarmingBot/1.0; "
            "+https://github.com/your-org/smart-farming)"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    DELAY_SECONDS = 1.5   # Polite crawl delay — respect robots.txt spirit

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _get(self, url: str, timeout: int = 20) -> Optional[BeautifulSoup]:
        try:
            resp = self.session.get(url, timeout=timeout)
            resp.raise_for_status()
            time.sleep(self.DELAY_SECONDS)
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.warning(f"GET failed [{url}]: {e}")
            return None


# ──────────────────────────────────────────────
# Source 1 — Vikaspedia (MeitY Government Portal)
# ──────────────────────────────────────────────

class VikaspediaScraper(BaseScraper):
    """
    Vikaspedia is an Indian government agriculture portal (Ministry of Electronics
    and IT). It publishes crop-wise disease management pages in English + Hindi.

    URL pattern: https://vikaspedia.in/agriculture/crop-production/
                   integrated-pest-managemen/crop-specific-ipm/<crop>
    """

    BASE = "https://vikaspedia.in"
    CROP_PATHS = {
        "wheat":     "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-wheat",
        "rice":      "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-rice",
        "tomato":    "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-tomato",
        "potato":    "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-potato",
        "onion":     "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-onion",
        "maize":     "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-maize",
        "cotton":    "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-cotton",
        "soybean":   "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-soybean",
        "sugarcane": "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-sugarcane",
        "barley":    "/agriculture/crop-production/integrated-pest-managemen/crop-specific-ipm/ipm-for-barley",
    }

    def scrape(self) -> List[Advisory]:
        advisories = []
        for crop, path in self.CROP_PATHS.items():
            url  = self.BASE + path
            soup = self._get(url)
            if not soup:
                continue

            advisories.extend(self._parse_page(soup, crop, url))
            logger.info(f"Vikaspedia | {crop}: {len(advisories)} total advisories so far")

        return advisories

    def _parse_page(self, soup: BeautifulSoup, crop: str, url: str) -> List[Advisory]:
        """
        Vikaspedia pages have disease tables:
          | Disease | Symptoms | Management |
        We extract all table rows with 3+ columns.
        """
        results = []
        tables  = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")
            for row in rows[1:]:   # Skip header
                cols = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cols) >= 3:
                    disease, symptoms, management = cols[0], cols[1], cols[2]
                    if len(disease) > 3 and len(management) > 10:
                        results.append(Advisory(
                            source="vikaspedia",
                            url=url,
                            crop=crop,
                            disease=disease,
                            symptoms=symptoms,
                            treatment=management,
                            scraped_at=datetime.utcnow().isoformat(),
                        ))

        # Also parse description paragraphs (some pages use lists instead of tables)
        headers = soup.find_all(["h3", "h4"])
        for h in headers:
            if any(word in h.text.lower() for word in ["blight", "rust", "rot", "spot", "wilt", "virus", "mite"]):
                disease_name = h.get_text(strip=True)
                content_parts = []
                sibling = h.find_next_sibling()
                while sibling and sibling.name not in ["h3", "h4"]:
                    content_parts.append(sibling.get_text(strip=True))
                    sibling = sibling.find_next_sibling()
                content = " ".join(content_parts)
                if len(content) > 30:
                    results.append(Advisory(
                        source="vikaspedia",
                        url=url,
                        crop=crop,
                        disease=disease_name,
                        symptoms="See full article",
                        treatment=content[:800],
                        scraped_at=datetime.utcnow().isoformat(),
                    ))

        return results


# ──────────────────────────────────────────────
# Source 2 — PlantVillage (Penn State)
# ──────────────────────────────────────────────

class PlantVillageScraper(BaseScraper):
    """
    PlantVillage (plantvillage.psu.edu) has factsheets for 300+ crop diseases
    with consistent structure: Overview | Symptoms | Management.

    These are the same classes used in their open dataset — aligns perfectly
    with our DISEASE_CLASSES taxonomy.
    """

    BASE = "https://plantvillage.psu.edu"
    DISEASE_URLS = {
        "tomato_late_blight":   "/topics/tomato/infos",
        "wheat_brown_rust":     "/topics/wheat/infos",
        "rice_leaf_blast":      "/topics/rice/infos",
        "potato_late_blight":   "/topics/potato/infos",
        "maize_common_rust":    "/topics/maize-corn/infos",
        "cotton_curl_virus":    "/topics/cotton/infos",
    }

    def scrape(self) -> List[Advisory]:
        advisories = []
        for disease_key, path in self.DISEASE_URLS.items():
            crop = disease_key.split("_")[0]
            url  = self.BASE + path
            soup = self._get(url)
            if not soup:
                continue

            # Find all disease factsheet links on the topic page
            links = soup.select("a[href*='/topics/']")
            for link in links[:10]:   # Cap at 10 per crop to avoid overloading
                factsheet_url = self.BASE + link["href"] if link["href"].startswith("/") else link["href"]
                advisory = self._parse_factsheet(factsheet_url, crop)
                if advisory:
                    advisories.append(advisory)

        return advisories

    def _parse_factsheet(self, url: str, crop: str) -> Optional[Advisory]:
        soup = self._get(url)
        if not soup:
            return None

        title    = soup.find("h1")
        disease  = title.get_text(strip=True) if title else "Unknown"

        # Extract sections
        sections = {}
        for heading in soup.find_all(["h2", "h3"]):
            key  = heading.get_text(strip=True).lower()
            body = []
            sib  = heading.find_next_sibling()
            while sib and sib.name not in ["h2", "h3"]:
                body.append(sib.get_text(strip=True))
                sib = sib.find_next_sibling()
            sections[key] = " ".join(body)

        symptoms   = sections.get("symptoms", sections.get("signs", "See article"))
        management = sections.get("management", sections.get("prevention and control", ""))

        if not management or len(management) < 20:
            return None

        return Advisory(
            source="plantvillage",
            url=url,
            crop=crop,
            disease=disease,
            symptoms=symptoms[:600],
            treatment=management[:800],
            scraped_at=datetime.utcnow().isoformat(),
        )


# ──────────────────────────────────────────────
# Source 3 — ICAR Advisory Bulletins
# ──────────────────────────────────────────────

class ICARScraper(BaseScraper):
    """
    ICAR (icar.org.in) publishes crop-specific disease advisories.
    We target their 'Crop Production' section for disease management tables.
    """

    ADVISORY_URLS = [
        "https://www.icar.org.in/content/crop-diseases",
        "https://www.icar.org.in/content/plant-protection",
    ]

    def scrape(self) -> List[Advisory]:
        advisories = []
        for url in self.ADVISORY_URLS:
            soup = self._get(url)
            if not soup:
                continue

            # ICAR pages use definition lists and paragraph blocks
            items = soup.select(".field-item, .views-field, article p")
            current_crop = "unknown"
            for item in items:
                text = item.get_text(strip=True)
                # Detect crop headings
                for crop in ["wheat", "rice", "tomato", "potato", "maize", "cotton",
                             "onion", "soybean", "sugarcane", "barley"]:
                    if crop.lower() in text.lower() and len(text) < 50:
                        current_crop = crop
                # Extract disease mentions
                if any(w in text.lower() for w in ["blight", "rust", "rot", "wilt",
                                                    "spot", "mildew", "mosaic", "smut"]):
                    if len(text) > 40:
                        advisories.append(Advisory(
                            source="icar",
                            url=url,
                            crop=current_crop,
                            disease="See text",
                            symptoms="",
                            treatment=text[:600],
                            scraped_at=datetime.utcnow().isoformat(),
                        ))

        logger.info(f"ICAR: scraped {len(advisories)} advisories")
        return advisories


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

class DiseaseAdvisoryScraper:
    """
    Orchestrates all scrapers, deduplicates results, and saves to JSON.

    Run once weekly via APScheduler (already in your requirements.txt):
        scheduler.add_job(scraper.run_and_save, 'interval', weeks=1)
    """

    def __init__(self, output_path: str = "data/raw/disease_advisories.json"):
        self.output_path = output_path
        self.scrapers = [
            VikaspediaScraper(),
            PlantVillageScraper(),
            ICARScraper(),
        ]

    def scrape_all(self) -> List[Advisory]:
        all_advisories = []
        seen_hashes    = set()

        for scraper in self.scrapers:
            name = scraper.__class__.__name__
            logger.info(f"Running {name}...")
            try:
                results = scraper.scrape()
                # Deduplicate by content hash
                for adv in results:
                    if adv.hash not in seen_hashes:
                        all_advisories.append(adv)
                        seen_hashes.add(adv.hash)
                logger.info(f"{name} → {len(results)} advisories ({len(all_advisories)} total unique)")
            except Exception as e:
                logger.error(f"{name} failed: {e}")

        return all_advisories

    def save(self, advisories: List[Advisory], path: str = None) -> str:
        path = path or self.output_path
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [asdict(a) for a in advisories]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(advisories)} advisories → {path}")
        return path

    def run_and_save(self) -> str:
        """Single entry point for scheduled jobs."""
        advisories = self.scrape_all()
        return self.save(advisories)


if __name__ == "__main__":
    scraper = DiseaseAdvisoryScraper()
    scraper.run_and_save()