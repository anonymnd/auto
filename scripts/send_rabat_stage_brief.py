#!/usr/bin/env python3
"""Send a daily email with recent public LinkedIn internship posts for Rabat."""

from __future__ import annotations

import argparse
import html
import os
import re
import smtplib
import ssl
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from html.parser import HTMLParser
from zoneinfo import ZoneInfo


SEARCH_URL = "https://html.duckduckgo.com/html/?{params}"
BING_RSS_URL = "https://www.bing.com/search?{params}"
BRAVE_URL = "https://search.brave.com/search?{params}"
YAHOO_URL = "https://search.yahoo.com/search?{params}"
ACTIVITY_RE = re.compile(r"activity(?:-|:|%3A)(\d+)", re.IGNORECASE)
LINKEDIN_URL_RE = re.compile(r"https?://[^\s\"<>]+linkedin\.com/[^\s\"<>]+", re.IGNORECASE)
RESULT_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)".*?</a>.*?'
    r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")

QUERIES = (
    'site:linkedin.com/posts Rabat stage juillet',
    'site:linkedin.com/posts Rabat internship July',
    'site:linkedin.com/posts Rabat PFA informatique',
    'site:linkedin.com/posts Rabat stage développeur',
    'site:linkedin.com/posts Rabat stage Laravel OR PHP OR Java OR React',
    'site:linkedin.com/posts Rabat stage data OR IA OR cybersécurité OR cloud OR DevOps',
)

IT_TERMS = (
    "informatique",
    "dévelop",
    "develop",
    "software",
    "data",
    " ia ",
    " ai ",
    "cyber",
    "cloud",
    "devops",
    "réseau",
    "reseau",
    "système",
    "system",
    "qa",
    "test",
    "support it",
    "full stack",
    "frontend",
    "backend",
)
JULY_TERMS = ("juillet", "july", "07/2026", "07-2026", "début juillet", "debut juillet")
EXCLUDED_TERMS = ("pfe", "fin d'études", "fin d’etudes", "end-of-studies", "end of studies")
CASABLANCA = ZoneInfo("Africa/Casablanca")
MONITORED_POSTS_PATH = os.path.join(os.path.dirname(__file__), "monitored_linkedin_posts.txt")
SCHEDULED_ALLOWED_HOURS = {8, 9}


@dataclass(frozen=True)
class Post:
    url: str
    snippet: str
    published_at: datetime


class MetaDescriptionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.descriptions: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = {key.lower(): value or "" for key, value in attrs}
        if values.get("name", "").lower() == "description" or values.get("property", "").lower() in {
            "og:description",
            "twitter:description",
        }:
            self.descriptions.append(values.get("content", ""))


def clean_text(value: str) -> str:
    return SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", value))).strip()


def unwrap_duckduckgo_url(url: str) -> str:
    value = html.unescape(url)
    parsed = urllib.parse.urlparse(value)
    if parsed.netloc.endswith("duckduckgo.com"):
        params = urllib.parse.parse_qs(parsed.query)
        value = params.get("uddg", [value])[0]
    return urllib.parse.unquote(value)


def unwrap_yahoo_url(url: str) -> str:
    value = html.unescape(url)
    match = re.search(r"/RU=(.*?)/RK=", value)
    return urllib.parse.unquote(match.group(1)) if match else value


def linkedin_activity_datetime(url: str) -> datetime | None:
    match = ACTIVITY_RE.search(url)
    if not match:
        return None
    activity_id = int(match.group(1))
    timestamp_ms = activity_id >> 22
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def make_post(url: str, snippet: str) -> Post | None:
    url = unwrap_duckduckgo_url(url)
    published_at = linkedin_activity_datetime(url)
    if "linkedin.com/" not in url or published_at is None:
        return None
    return Post(url=url, snippet=clean_text(snippet), published_at=published_at)


def search_duckduckgo(query: str) -> list[Post]:
    params = urllib.parse.urlencode({"q": query, "df": "w"})
    request = urllib.request.Request(
        SEARCH_URL.format(params=params),
        headers={"User-Agent": "Mozilla/5.0 (compatible; RabatStageBrief/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8", errors="replace")

    posts: list[Post] = []
    for match in RESULT_RE.finditer(page):
        post = make_post(match.group("url"), match.group("snippet"))
        if post is not None:
            posts.append(post)
    return posts


def search_bing(query: str) -> list[Post]:
    params = urllib.parse.urlencode({"q": query, "format": "rss"})
    request = urllib.request.Request(
        BING_RSS_URL.format(params=params),
        headers={"User-Agent": "Mozilla/5.0 (compatible; RabatStageBrief/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())

    posts: list[Post] = []
    for item in root.findall(".//item"):
        post = make_post(item.findtext("link", ""), item.findtext("description", ""))
        if post is not None:
            posts.append(post)
    return posts


def search_html_urls(query: str, endpoint: str, parameter: str, unwrap=lambda url: url) -> list[Post]:
    params = urllib.parse.urlencode({parameter: query})
    request = urllib.request.Request(
        endpoint.format(params=params),
        headers={"User-Agent": "Mozilla/5.0 (compatible; RabatStageBrief/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        page = html.unescape(response.read().decode("utf-8", errors="replace"))

    posts: list[Post] = []
    for raw_url in LINKEDIN_URL_RE.findall(page):
        post = make_post(unwrap(raw_url), "")
        if post is not None:
            posts.append(post)
    return posts


def search_brave(query: str) -> list[Post]:
    return search_html_urls(query, BRAVE_URL, "q")


def search_yahoo(query: str) -> list[Post]:
    return search_html_urls(query, YAHOO_URL, "p", unwrap_yahoo_url)


def monitored_posts() -> list[Post]:
    if not os.path.exists(MONITORED_POSTS_PATH):
        return []
    posts: list[Post] = []
    with open(MONITORED_POSTS_PATH, encoding="utf-8") as source:
        for line in source:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            post = make_post(url, "")
            if post is not None:
                posts.append(post)
    return posts


def fetch_linkedin_text(post: Post) -> str:
    request = urllib.request.Request(
        post.url,
        headers={
            "Accept-Language": "fr,en;q=0.8",
            "User-Agent": "Mozilla/5.0 (compatible; RabatStageBrief/1.0)",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            page = response.read().decode("utf-8", errors="replace")
        parser = MetaDescriptionParser()
        parser.feed(page)
        descriptions = [clean_text(value) for value in parser.descriptions if clean_text(value)]
        return min(descriptions, key=len) if descriptions else post.snippet
    except Exception as exc:
        print(f"LinkedIn page warning for {post.url}: {exc}", file=sys.stderr)
        return post.snippet


def is_relevant(post: Post, now: datetime) -> bool:
    age = now - post.published_at
    text = f" {post.snippet.lower()} "
    return (
        timedelta(0) <= age <= timedelta(days=7)
        and "rabat" in text
        and any(term in text for term in IT_TERMS)
        and any(term in text for term in JULY_TERMS)
        and not any(term in text for term in EXCLUDED_TERMS)
    )


def collect_posts(now: datetime) -> list[Post]:
    discovered: dict[str, Post] = {}
    errors: list[str] = []
    for post in monitored_posts():
        activity_id = ACTIVITY_RE.search(post.url)
        discovered[activity_id.group(1) if activity_id else post.url] = post
    for index, query in enumerate(QUERIES):
        engines = (search_duckduckgo, search_bing, search_yahoo, search_brave) if index == 0 else (
            search_duckduckgo,
            search_bing,
            search_yahoo,
        )
        for engine in engines:
            try:
                for post in engine(query):
                    if now - post.published_at <= timedelta(days=7):
                        activity_id = ACTIVITY_RE.search(post.url)
                        discovered[activity_id.group(1) if activity_id else post.url] = post
            except Exception as exc:  # Continue so one failed query does not cancel the brief.
                errors.append(f"{engine.__name__} | {query}: {exc}")
    if errors:
        print("Search warnings:", *errors, sep="\n- ", file=sys.stderr)
    unique: dict[str, Post] = {}
    for key, post in discovered.items():
        enriched = Post(url=post.url, snippet=fetch_linkedin_text(post), published_at=post.published_at)
        if is_relevant(enriched, now):
            unique[key] = enriched
    return sorted(unique.values(), key=lambda post: post.published_at, reverse=True)


def build_brief(posts: list[Post], now: datetime) -> str:
    lines = [
        f"Brief stages informatique Rabat - {now.astimezone().strftime('%d/%m/%Y')}",
        "",
        "Filtres: stage classique / internship / PFA, début juillet, posts LinkedIn publics publiés depuis 7 jours maximum, hors PFE.",
        "",
    ]
    if not posts:
        lines.append("Aucun nouveau post LinkedIn public conforme trouvé aujourd'hui.")
        return "\n".join(lines)

    lines.append(f"{len(posts)} opportunité(s) récente(s) trouvée(s):")
    lines.append("")
    for index, post in enumerate(posts, start=1):
        lines.extend(
            [
                f"{index}. Publication LinkedIn du {post.published_at.astimezone().strftime('%d/%m/%Y %H:%M')}",
                f"   {post.snippet}",
                f"   {post.url}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def send_email(subject: str, body: str) -> None:
    sender = os.environ["GMAIL_SENDER"]
    recipient = os.environ.get("BRIEF_RECIPIENT", sender)
    app_password = os.environ["GMAIL_APP_PASSWORD"].replace(" ", "")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(sender, app_password)
        smtp.send_message(message)


def print_brief(body: str) -> None:
    try:
        print(body)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((body + "\n").encode("utf-8", errors="replace"))


def should_send_scheduled(local_now: datetime) -> bool:
    return local_now.hour in SCHEDULED_ALLOWED_HOURS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Send only during the 08:00 hour in Casablanca. Manual runs send immediately.",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    local_now = now.astimezone(CASABLANCA)
    if args.scheduled and not should_send_scheduled(local_now):
        print(
            "Skipping scheduled run: "
            f"current Casablanca time is {local_now:%H:%M}; "
            "allowed scheduled hours are 08:00-09:59."
        )
        return

    posts = collect_posts(now)
    body = build_brief(posts, now)
    send_email(f"Brief stages informatique Rabat - {now.astimezone().strftime('%d/%m/%Y')}", body)
    print_brief(body)


if __name__ == "__main__":
    main()
