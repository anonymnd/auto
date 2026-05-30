#!/usr/bin/env python3
"""Send a daily email with recent public LinkedIn internship posts for Rabat."""

from __future__ import annotations

import html
import os
import re
import smtplib
import ssl
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage


SEARCH_URL = "https://html.duckduckgo.com/html/?{params}"
ACTIVITY_RE = re.compile(r"activity-(\d+)")
RESULT_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)".*?</a>.*?'
    r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")

QUERIES = (
    'site:linkedin.com/posts Rabat stage informatique juillet -PFE',
    'site:linkedin.com/posts Rabat internship informatique July -PFE',
    'site:linkedin.com/posts Rabat PFA informatique juillet -PFE',
    'site:linkedin.com/posts Rabat stage développeur data IA cybersécurité juillet -PFE',
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


@dataclass(frozen=True)
class Post:
    url: str
    snippet: str
    published_at: datetime


def clean_text(value: str) -> str:
    return SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", value))).strip()


def unwrap_duckduckgo_url(url: str) -> str:
    value = html.unescape(url)
    parsed = urllib.parse.urlparse(value)
    if parsed.netloc.endswith("duckduckgo.com"):
        params = urllib.parse.parse_qs(parsed.query)
        value = params.get("uddg", [value])[0]
    return urllib.parse.unquote(value)


def linkedin_activity_datetime(url: str) -> datetime | None:
    match = ACTIVITY_RE.search(url)
    if not match:
        return None
    activity_id = int(match.group(1))
    timestamp_ms = activity_id >> 22
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def search(query: str) -> list[Post]:
    params = urllib.parse.urlencode({"q": query, "df": "w"})
    request = urllib.request.Request(
        SEARCH_URL.format(params=params),
        headers={"User-Agent": "Mozilla/5.0 (compatible; RabatStageBrief/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8", errors="replace")

    posts: list[Post] = []
    for match in RESULT_RE.finditer(page):
        url = unwrap_duckduckgo_url(match.group("url"))
        published_at = linkedin_activity_datetime(url)
        if "linkedin.com/posts/" not in url or published_at is None:
            continue
        posts.append(Post(url=url, snippet=clean_text(match.group("snippet")), published_at=published_at))
    return posts


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
    unique: dict[str, Post] = {}
    errors: list[str] = []
    for query in QUERIES:
        try:
            for post in search(query):
                if is_relevant(post, now):
                    unique[post.url] = post
        except Exception as exc:  # Continue so one failed query does not cancel the brief.
            errors.append(f"{query}: {exc}")
    if errors:
        print("Search warnings:", *errors, sep="\n- ", file=sys.stderr)
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


def main() -> None:
    now = datetime.now(timezone.utc)
    posts = collect_posts(now)
    body = build_brief(posts, now)
    send_email(f"Brief stages informatique Rabat - {now.astimezone().strftime('%d/%m/%Y')}", body)
    print(body)


if __name__ == "__main__":
    main()
