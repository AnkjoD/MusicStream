#!/usr/bin/env python3
"""
Eventsim-compatible music streaming event producer.
Generates synthetic music listening events and sends them to Kafka.
Mimics the data shape of the original Scala eventsim project.
All config is via environment variables — no source code modification needed.
"""

import json
import os
import random
import time
import uuid
import math
from datetime import datetime, timezone
from typing import Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError

# ── Config from env ──────────────────────────────────────────────────────────
KAFKA_BROKER   = os.environ.get("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC    = os.environ.get("KAFKA_TOPIC", "eventsim")
NUM_USERS      = int(os.environ.get("NUM_USERS", "200"))
GROWTH_RATE    = float(os.environ.get("GROWTH_RATE", "0.0"))
EVENT_DELAY_MS = float(os.environ.get("EVENT_DELAY_MS", "200"))   # ms between events
REAL_TIME      = os.environ.get("REAL_TIME", "true").lower() == "true"

# ── Static Data ──────────────────────────────────────────────────────────────
PAGES_LOGGED_IN = [
    "NextSong", "NextSong", "NextSong", "NextSong",  # weighted more
    "Home", "About", "Settings", "Logout",
    "Downgrade", "Upgrade", "Help", "Error",
    "Add to Playlist", "Add Friend", "Thumbs Up", "Thumbs Down",
    "Roll Advert",
]
PAGES_LOGGED_OUT = ["Home", "Login", "Register", "About", "Error"]
LEVELS = ["free", "paid"]
GENDERS = ["M", "F"]
METHODS = ["GET", "PUT", "DELETE"]
AUTH_STATES = ["Logged In", "Logged Out", "Guest"]

LOCATIONS = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
    "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
    "Dallas, TX", "San Jose, CA", "Austin, TX", "Jacksonville, FL",
    "San Francisco, CA", "Columbus, OH", "Indianapolis, IN", "Seattle, WA",
]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
]
FIRST_NAMES = [
    "James","John","Robert","Michael","William","David","Richard","Joseph",
    "Thomas","Charles","Mary","Patricia","Jennifer","Linda","Barbara",
    "Elizabeth","Susan","Jessica","Sarah","Karen","Emma","Olivia","Ava",
    "Isabella","Sophia","Mia","Charlotte","Amelia","Harper","Evelyn",
]
LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Wilson","Anderson","Taylor","Thomas","Jackson","White","Harris",
    "Martin","Thompson","Moore","Young","Allen","King","Wright","Scott",
]
ARTISTS = [
    "Coldplay","Radiohead","The Beatles","Pink Floyd","Led Zeppelin",
    "Arctic Monkeys","Tame Impala","Mac DeMarco","Bon Iver","Sufjan Stevens",
    "Taylor Swift","Billie Eilish","The Weeknd","Drake","Kendrick Lamar",
    "Frank Ocean","Tyler, the Creator","SZA","H.E.R.","Daniel Caesar",
    "Daft Punk","Boards of Canada","Brian Eno","Aphex Twin","Four Tet",
    "Massive Attack","Portishead","Burial","SOHN","James Blake",
]
SONGS: list[tuple[str, str, float]] = [
    ("Coldplay", "Yellow", 269.0),
    ("Coldplay", "The Scientist", 309.0),
    ("Radiohead", "Creep", 238.0),
    ("Radiohead", "Karma Police", 264.0),
    ("The Beatles", "Hey Jude", 431.0),
    ("The Beatles", "Let It Be", 243.0),
    ("Pink Floyd", "Comfortably Numb", 382.0),
    ("Pink Floyd", "Wish You Were Here", 334.0),
    ("Led Zeppelin", "Stairway to Heaven", 482.0),
    ("Arctic Monkeys", "505", 254.0),
    ("Arctic Monkeys", "Do I Wanna Know?", 272.0),
    ("Tame Impala", "The Less I Know the Better", 216.0),
    ("Tame Impala", "Feels Like We Only Go Backwards", 200.0),
    ("Taylor Swift", "Anti-Hero", 200.0),
    ("Billie Eilish", "Bad Guy", 194.0),
    ("The Weeknd", "Blinding Lights", 200.0),
    ("Drake", "God's Plan", 198.0),
    ("Kendrick Lamar", "HUMBLE.", 177.0),
    ("Frank Ocean", "Thinking Bout You", 200.0),
    ("Daft Punk", "Get Lucky", 369.0),
    ("Daft Punk", "Around the World", 428.0),
    ("Aphex Twin", "Windowlicker", 397.0),
    ("James Blake", "Limit to Your Love", 249.0),
    ("Mac DeMarco", "Chamber of Reflection", 236.0),
    ("Bon Iver", "Skinny Love", 222.0),
    ("Massive Attack", "Teardrop", 330.0),
]


# ── User Model ───────────────────────────────────────────────────────────────
class SimUser:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.session_id = random.randint(1, 100000)
        self.first_name = random.choice(FIRST_NAMES)
        self.last_name = random.choice(LAST_NAMES)
        self.gender = random.choice(GENDERS)
        self.level = random.choice(LEVELS)
        self.location = random.choice(LOCATIONS)
        self.user_agent = random.choice(USER_AGENTS)
        self.registration = int(datetime.now(timezone.utc).timestamp() * 1000) - random.randint(0, 31536000000)
        self.auth = "Logged In"
        self.item_in_session = 0
        self.current_song: Optional[tuple] = None
        self._next_event_offset = random.uniform(0, 10)   # seconds

    def next_event_ts(self) -> int:
        return int((time.time() + self._next_event_offset) * 1000)

    def advance(self) -> dict:
        """Generate next event and advance state."""
        ts = int(time.time() * 1000)
        self.item_in_session += 1

        # Pick page
        if self.auth == "Logged In":
            page = random.choices(
                PAGES_LOGGED_IN,
                weights=[4,4,4,4, 2,1,1,1, 0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5, 0.5],
                k=1
            )[0]
        else:
            page = random.choice(PAGES_LOGGED_OUT)

        # Build event
        event = {
            "ts": ts,
            "userId": self.user_id if self.auth == "Logged In" else "",
            "sessionId": self.session_id,
            "page": page,
            "auth": self.auth,
            "method": "GET" if page in ("Home","About","Settings","Help","Error") else "PUT",
            "status": 200,
            "level": self.level,
            "itemInSession": self.item_in_session,
            "firstName": self.first_name,
            "lastName": self.last_name,
            "gender": self.gender,
            "registration": self.registration,
            "location": self.location,
            "userAgent": self.user_agent,
        }

        # Handle specific pages
        if page == "NextSong":
            self.current_song = random.choice(SONGS)
            event["artist"] = self.current_song[0]
            event["song"]   = self.current_song[1]
            event["length"] = self.current_song[2]
        elif page == "Logout":
            self.auth = "Logged Out"
            self.session_id = random.randint(1, 100000)
            self.item_in_session = 0
        elif page == "Login":
            self.auth = "Logged In"
        elif page == "Upgrade":
            self.level = "paid"
        elif page == "Downgrade":
            self.level = "free"

        # Random session end
        if random.random() < 0.01:
            self.session_id = random.randint(1, 100000)
            self.item_in_session = 0

        # Schedule next event
        self._next_event_offset = random.expovariate(1 / 3.0)  # avg 3s between events

        return event


# ── Kafka Setup ──────────────────────────────────────────────────────────────
def create_producer(broker: str, retries: int = 10) -> KafkaProducer:
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=broker,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks=1,
                linger_ms=5,
                batch_size=16384,
                compression_type="gzip",
            )
            print(f"[eventsim] Connected to Kafka at {broker}")
            return producer
        except KafkaError as e:
            wait = 2 ** attempt
            print(f"[eventsim] Kafka not ready (attempt {attempt+1}/{retries}), retrying in {wait}s... {e}")
            time.sleep(wait)
    raise RuntimeError(f"Could not connect to Kafka at {broker} after {retries} attempts")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"[eventsim] Starting Python event producer")
    print(f"  broker={KAFKA_BROKER} topic={KAFKA_TOPIC} users={NUM_USERS} delay={EVENT_DELAY_MS}ms")

    producer = create_producer(KAFKA_BROKER)

    # Init users
    users = [SimUser(uid) for uid in range(1, NUM_USERS + 1)]
    sent = 0
    errors = 0

    print(f"[eventsim] Producing events... (Ctrl+C to stop)")
    try:
        while True:
            # Round-robin through users to generate events
            user = random.choice(users)
            event = user.advance()

            future = producer.send(KAFKA_TOPIC, event)
            try:
                future.get(timeout=5)
                sent += 1
                if sent % 1000 == 0:
                    print(f"[eventsim] Sent {sent} events ({errors} errors) | "
                          f"last page={event['page']} user={event['userId']}")
            except KafkaError as e:
                errors += 1
                print(f"[eventsim] Send error: {e}")

            if REAL_TIME:
                time.sleep(EVENT_DELAY_MS / 1000.0)

    except KeyboardInterrupt:
        print(f"\n[eventsim] Stopping. Total sent={sent} errors={errors}")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
