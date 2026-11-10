import os
import sys
import json
import time
import select
import difflib
import termios
import threading
import subprocess
from datetime import datetime
from urllib.parse import quote_plus
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

load_dotenv()
OWNER = os.getenv("OWNER_NAME", "sir")
# gemini-2.5-flash reasons and follows multi-step instructions far better than
# flash-lite, and is still free-tier + fast. flash-lite stays as a fallback.
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# When the primary model is overloaded (503), fall back to these in order.
# All are free-tier. The conversation history is carried across. The lighter
# models (flash-lite / 2.0) are usually less congested, so they make good
# fallbacks when 2.5-flash is swamped.
FALLBACK_MODELS = [MODEL] + [
    m for m in ("gemini-2.5-flash", "gemini-2.0-flash",
                "gemini-2.5-flash-lite", "gemini-2.0-flash-lite")
    if m != MODEL
]
# HTTP codes worth retrying: overloaded / rate-limited / transient server error.
RETRYABLE_CODES = {429, 500, 503}

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# Small helper to run shell commands safely
# ---------------------------------------------------------------------------

def _run(cmd, timeout=10):
    """Run a command list, return (ok, output_text)."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        text = (out.stdout or "").strip()
        err = (out.stderr or "").strip()
        return (out.returncode == 0, text if text else err)
    except Exception as e:
        return (False, str(e))


def _am(args, timeout=12):
    """Run `am start <args>`. Android's `am` lies with exit codes, so judge
    success by the output: it prints 'Starting:' on success and 'Error:'/an
    exception on failure. Returns (ok, output_text)."""
    try:
        r = subprocess.run(
            ["am", "start"] + args, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        blob = ((r.stdout or "") + (r.stderr or "")).strip()
        ok = ("Error" not in blob) and ("Exception" not in blob)
        return (ok, blob)
    except Exception as e:
        return (False, str(e))


# ---------------------------------------------------------------------------
# TOOLS — phone control functions Gemini can call.
# Every function needs a docstring describing args + behavior.
# ---------------------------------------------------------------------------

# --- Device state & hardware ---------------------------------------------

def set_flashlight(on: bool) -> str:
    """Turn the phone's flashlight (torch) on or off.

    Args:
        on: True to turn the flashlight ON, False to turn it OFF.

    Returns:
        Confirmation message.
    """
    _run(["termux-torch", "on" if on else "off"])
    return f"Flashlight is now {'ON' if on else 'OFF'}."


def get_battery_status() -> str:
    """Get the phone's current battery percentage and charging status.

    Returns:
        A short human-readable summary of battery level and status.
    """
    ok, out = _run(["termux-battery-status"], timeout=5)
    if not ok:
        return f"Could not read battery: {out}"
    try:
        data = json.loads(out)
        pct = data.get("percentage", "?")
        status = data.get("status", "unknown").lower()
        return f"Battery at {pct}%, status: {status}."
    except Exception as e:
        return f"Could not parse battery: {e}"


def get_volume() -> str:
    """Get current volume levels for all audio streams (music, ringtone, etc.).

    Returns:
        Volume info as a string.
    """
    ok, out = _run(["termux-volume"], timeout=5)
    return out if ok else f"Could not read volume: {out}"


def set_volume(stream: str, level: int) -> str:
    """Set the volume of an audio stream.

    Args:
        stream: One of "music", "call", "ring", "alarm", "notification", "system".
        level: Volume level, typically 0 to 15 (0 = silent).

    Returns:
        Confirmation message.
    """
    ok, out = _run(["termux-volume", stream, str(level)])
    if ok:
        return f"{stream.capitalize()} volume set to {level}."
    return f"Could not set volume: {out}"


def set_auto_brightness(on: bool = True) -> str:
    """Turn the phone's automatic (adaptive) brightness ON or OFF. Use this when
    the user asks for 'auto brightness', 'adaptive brightness', or to stop the
    screen dimming/brightening on its own.

    Args:
        on: True to enable auto/adaptive brightness, False to switch to manual.

    Returns:
        Confirmation message.
    """
    # 1 = automatic, 0 = manual.
    ok, out = _run(["termux-brightness", "auto"]) if on else (True, "")
    # Best-effort direct write too (works if Termux:API holds WRITE_SETTINGS).
    _run(["settings", "put", "system", "screen_brightness_mode", "1" if on else "0"])
    if on:
        return ("Automatic brightness is back on, sir. The screen will adjust to "
                "the light on its own now.")
    return "Automatic brightness is off; brightness is manual now."


def set_brightness(percent: int) -> str:
    """Set the screen brightness to a fixed level. NOTE: like any manual
    brightness change on Android, this switches OFF automatic/adaptive brightness
    until it is turned back on (see set_auto_brightness).

    Args:
        percent: Brightness from 0 (dimmest) to 100 (brightest).

    Returns:
        Confirmation message.
    """
    percent = max(0, min(100, percent))
    value = round(percent * 255 / 100)
    _run(["termux-brightness", str(value)])

    # termux-brightness can exit 0 even when the Termux:API app actually FAILED
    # to apply the change (missing "Modify system settings" permission, or the
    # helper was killed by Android's phantom-process monitor). So don't trust the
    # exit code - read the real brightness back and compare.
    ok, current = _run(["settings", "get", "system", "screen_brightness"], timeout=5)
    if ok and current.strip().lstrip("-").isdigit():
        actual = int(current.strip())
        if abs(actual - value) <= 8:            # allow rounding slack
            return (f"Brightness set to {percent}%. (This turns off auto "
                    f"brightness - say 'turn on auto brightness' to restore it.)")
        return ("I tried, but the brightness didn't actually change. The "
                "Termux:API app needs the 'Modify system settings' permission - "
                "grant that in Settings and try again, sir.")
    # Couldn't verify - be honest rather than claim a false success.
    return ("I sent the brightness change, but couldn't confirm it applied. If "
            "the screen didn't change, grant the Termux:API app the 'Modify "
            "system settings' permission.")


def vibrate(milliseconds: int = 500) -> str:
    """Vibrate the phone.

    Args:
        milliseconds: How long to vibrate, in milliseconds. Default 500.

    Returns:
        Confirmation message.
    """
    _run(["termux-vibrate", "-d", str(milliseconds), "-f"])
    return f"Vibrated for {milliseconds} ms."


def get_time_and_date() -> str:
    """Get the current local time and date.

    Returns:
        The current time and date as a readable string.
    """
    now = datetime.now()
    return now.strftime("It is %I:%M %p on %A, %d %B %Y.")


# --- Clipboard & notifications -------------------------------------------

def get_clipboard() -> str:
    """Read whatever text is currently on the phone's clipboard.

    Returns:
        The clipboard contents.
    """
    ok, out = _run(["termux-clipboard-get"], timeout=5)
    return out if out else "The clipboard is empty."


def set_clipboard(text: str) -> str:
    """Copy text onto the phone's clipboard.

    Args:
        text: The text to place on the clipboard.

    Returns:
        Confirmation message.
    """
    try:
        subprocess.run(["termux-clipboard-set"], input=text, text=True, timeout=5)
        return "Copied to clipboard."
    except Exception as e:
        return f"Could not set clipboard: {e}"


def show_notification(title: str, message: str) -> str:
    """Post a notification to the phone's notification shade.

    Args:
        title: The notification title.
        message: The notification body text.

    Returns:
        Confirmation message.
    """
    ok, out = _run(["termux-notification", "--title", title, "--content", message])
    return "Notification posted." if ok else f"Could not post: {out}"


# --- Apps, web, intents (Android activity manager) -----------------------

# Chrome's package + main activity. Targeting this component makes a URL open
# in Chrome specifically, instead of whatever the default browser happens to be.
CHROME_PKG = "com.android.chrome"
CHROME_ACTIVITY = "com.google.android.apps.chrome.Main"


def _open_first(uris):
    """Try opening each URI in turn via a VIEW intent; stop at the first that
    launches. Lets us prefer an app's deep link (instagram://...) and fall back
    to the equivalent web URL. Returns (ok, output_text)."""
    out = ""
    for u in uris:
        ok, out = _am(["-a", "android.intent.action.VIEW", "-d", u])
        if ok:
            return (True, out)
    return (False, out)


def _open_in_chrome(url):
    """Open a URL in Chrome. Falls back to the default browser if Chrome isn't
    installed/available. Returns (ok, output_text)."""
    ok, out = _am(["-n", f"{CHROME_PKG}/{CHROME_ACTIVITY}",
                   "-a", "android.intent.action.VIEW", "-d", url])
    if ok:
        return (True, out)
    # Chrome not available — hand off to whatever browser is the default.
    return _am(["-a", "android.intent.action.VIEW", "-d", url])


# Known spoken-name -> package. Aliases help us jump straight to the right app
# for popular apps whose package name isn't obvious from the spoken name
# (e.g. "mx player" -> com.mxtech.videoplayer.ad). Anything NOT here is still
# found by matching against the phone's real installed-package list (see
# _resolve_app), so this list only needs to cover the tricky ones.
APP_PACKAGES = {
    # Google / system
    "youtube": "com.google.android.youtube",
    "youtube music": "com.google.android.apps.youtube.music",
    "chrome": "com.android.chrome",
    "gmail": "com.google.android.gm",
    "maps": "com.google.android.apps.maps",
    "google maps": "com.google.android.apps.maps",
    "google": "com.google.android.googlequicksearchbox",
    "photos": "com.google.android.apps.photos",
    "google photos": "com.google.android.apps.photos",
    "calendar": "com.google.android.calendar",
    "drive": "com.google.android.apps.docs",
    "google drive": "com.google.android.apps.docs",
    "meet": "com.google.android.apps.tachyon",
    "google meet": "com.google.android.apps.tachyon",
    "play store": "com.android.vending",
    "playstore": "com.android.vending",
    "play": "com.android.vending",
    "google play": "com.android.vending",
    "google play store": "com.android.vending",
    # Messaging / social
    "whatsapp": "com.whatsapp",
    "whatsapp business": "com.whatsapp.w4b",
    "instagram": "com.instagram.android",
    "insta": "com.instagram.android",
    "facebook": "com.facebook.katana",
    "fb": "com.facebook.katana",
    "messenger": "com.facebook.orca",
    "telegram": "org.telegram.messenger",
    "snapchat": "com.snapchat.android",
    "threads": "com.instagram.barcelona",
    "x": "com.twitter.android",
    "twitter": "com.twitter.android",
    "linkedin": "com.linkedin.android",
    "truecaller": "com.truecaller",
    "pinterest": "com.pinterest",
    # Shopping / payments (India)
    "flipkart": "com.flipkart.android",
    "amazon": "in.amazon.mShop.android.shopping",
    "myntra": "com.myntra.android",
    "meesho": "com.meesho.supply",
    "paytm": "net.one97.paytm",
    "phonepe": "com.phonepe.app",
    "google pay": "com.google.android.apps.nbu.paisa.user",
    "gpay": "com.google.android.apps.nbu.paisa.user",
    "cred": "com.dreamplug.androidapp",
    # Streaming / music
    "netflix": "com.netflix.mediaclient",
    "prime video": "com.amazon.avod.thirdpartyclient",
    "amazon prime": "com.amazon.avod.thirdpartyclient",
    "hotstar": "in.startv.hotstar",
    "jiohotstar": "in.startv.hotstar",
    "disney hotstar": "in.startv.hotstar",
    "spotify": "com.spotify.music",
    "gaana": "com.gaana",
    "wynk": "com.bsbportal.music",
    "jiosaavn": "com.jio.media.jiobeats",
    "saavn": "com.jio.media.jiobeats",
    "mx player": "com.mxtech.videoplayer.ad",
    "mxplayer": "com.mxtech.videoplayer.ad",
    "vlc": "org.videolan.vlc",
    # Food / travel
    "zomato": "com.application.zomato",
    "swiggy": "in.swiggy.android",
    "uber": "com.ubercab",
    "ola": "com.olacabs.customer",
    "rapido": "com.rapido.passenger",
    # Jio
    "myjio": "com.jio.myjio",
    "my jio": "com.jio.myjio",
    "jio": "com.jio.myjio",
    "jiocinema": "com.jio.media.ondemand",
    # Microsoft / work
    "outlook": "com.microsoft.office.outlook",
    "teams": "com.microsoft.teams",
    "microsoft teams": "com.microsoft.teams",
    "word": "com.microsoft.office.word",
    "excel": "com.microsoft.office.excel",
    "onenote": "com.microsoft.office.onenote",
    "zoom": "us.zoom.videomeetings",
    "canva": "com.canva.editor",
}

# Apps best opened via a system action intent (no package needed).
APP_ACTIONS = {
    "settings": ["-a", "android.settings.SETTINGS"],
    "clock": ["-a", "android.intent.action.SHOW_ALARMS"],
    "alarm": ["-a", "android.intent.action.SHOW_ALARMS"],
    "alarms": ["-a", "android.intent.action.SHOW_ALARMS"],
    "camera": ["-a", "android.media.action.STILL_IMAGE_CAMERA"],
    "calculator": ["-a", "android.intent.action.MAIN",
                   "-c", "android.intent.category.APP_CALCULATOR"],
    "contacts": ["-a", "android.intent.action.MAIN",
                 "-c", "android.intent.category.APP_CONTACTS"],
    "email": ["-a", "android.intent.action.MAIN",
              "-c", "android.intent.category.APP_EMAIL"],
    "browser": ["-a", "android.intent.action.MAIN",
                "-c", "android.intent.category.APP_BROWSER"],
    "music": ["-a", "android.intent.action.MAIN",
              "-c", "android.intent.category.APP_MUSIC"],
    "gallery": ["-a", "android.intent.action.MAIN",
                "-c", "android.intent.category.APP_GALLERY"],
}


def _launch_package(pkg: str) -> bool:
    """Resolve a package's launcher activity and start it via `am`.
    Uses --user 0 because Termux can't query the default 'current user' (-2)."""
    ok, out = _run(
        ["cmd", "package", "resolve-activity", "--brief", "--user", "0", pkg],
        timeout=8,
    )
    if ok and out and "Exception" not in out:
        comp = out.splitlines()[-1].strip()
        if "/" in comp:
            started, _ = _am(["-n", comp])
            if started:
                return True
    # Fallback: monkey can launch a package by its LAUNCHER category directly.
    ok, out = _run(
        ["monkey", "-p", pkg, "-c", "android.intent.category.LAUNCHER", "1"],
        timeout=8,
    )
    return ok and "No activities found" not in out and "Error" not in out


# Cache of installed package names, refreshed lazily.
_pkgs_cache = {"data": None, "at": 0.0}


def _installed_packages(max_age=300):
    """All installed package names on the phone (via `pm list packages`), cached.
    Lets us open ANY installed app by matching the spoken name to a real package,
    not just the ones we hardcoded."""
    now = time.monotonic()
    if _pkgs_cache["data"] is not None and now - _pkgs_cache["at"] < max_age:
        return _pkgs_cache["data"]
    ok, out = _run(["pm", "list", "packages"], timeout=12)
    pkgs = []
    if ok:
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("package:"):
                pkgs.append(line[len("package:"):])
    _pkgs_cache["data"] = pkgs
    _pkgs_cache["at"] = now
    return pkgs


def _norm_app(name):
    """Normalise a spoken app name: lowercase, drop filler words/punctuation."""
    s = name.strip().lower()
    for junk in ("please", "the app", " app", "open ", "just ", "my "):
        s = s.replace(junk, " ")
    s = "".join(ch if ch.isalnum() or ch == " " else " " for ch in s)
    return " ".join(s.split())


def _resolve_app(name):
    """Turn a spoken app name into an installed package to launch. Strategy:
    1) alias table (handles tricky names),
    2) exact/'contains' match against real installed package names,
    3) fuzzy match against the words inside installed package names.
    Returns a package string, or None."""
    key = _norm_app(name)
    if not key:
        return None

    installed = set(_installed_packages())

    # 1) Alias table - prefer it, but only if that package is actually installed
    #    (fall back to trying it anyway if we couldn't read the package list).
    for alias in (key, key.replace(" ", "")):
        pkg = APP_PACKAGES.get(alias)
        if pkg and (pkg in installed or not installed):
            return pkg

    if not installed:
        return APP_PACKAGES.get(key)  # can't match further without the list

    compact = key.replace(" ", "")

    # 2) Package name literally contains the spoken word(s).
    #    e.g. "flipkart" -> com.flipkart.android, "truecaller" -> com.truecaller
    best_contains = None
    for pkg in installed:
        p = pkg.lower()
        if compact and compact in p.replace(".", ""):
            # Prefer the shortest package (usually the main app, not a plugin).
            if best_contains is None or len(pkg) < len(best_contains):
                best_contains = pkg
    if best_contains:
        return best_contains

    # 3) Fuzzy match against the segments of each package name.
    target = compact
    best_pkg, best_score = None, 0.0
    for pkg in installed:
        for seg in pkg.lower().split("."):
            if seg in ("com", "android", "app", "apps", "org", "net", "in", "co"):
                continue
            score = difflib.SequenceMatcher(None, target, seg).ratio()
            if score > best_score:
                best_score, best_pkg = score, pkg
    if best_pkg and best_score >= 0.72:
        return best_pkg
    return None


def open_app(name: str) -> str:
    """Open ANY installed app on the phone by the name the user calls it - e.g.
    "WhatsApp", "MX Player", "Flipkart", "Play Store", "MyJio", "Settings".
    It matches against the phone's actual installed apps, so it is not limited to
    a fixed list. Use the name as the user says it; don't demand an exact package.

    Args:
        name: The app's spoken/common name.

    Returns:
        Confirmation message, or an error if the app truly can't be found.
    """
    key = _norm_app(name)

    # 1) System apps via action intent (most reliable on this phone).
    if key in APP_ACTIONS:
        ok, _ = _am(APP_ACTIONS[key])
        if ok:
            return f"Opening {name}."

    # 2) Resolve to a real installed package and launch it.
    pkg = _resolve_app(name)
    if pkg and _launch_package(pkg):
        return f"Opening {name}."

    return (f"I couldn't find '{name}' among your installed apps, sir. Could you "
            f"say it the way it appears in your app drawer?")


def open_url(url: str) -> str:
    """Open a web page or link in Chrome.

    Args:
        url: The full URL (include https://).

    Returns:
        Confirmation message.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    ok, out = _open_in_chrome(url)
    return f"Opening {url} in Chrome." if ok else f"Could not open link: {out}"


def open_in_chrome(query_or_url: str) -> str:
    """Open Chrome, either to a specific web page or to Google results for a
    search phrase. Use this whenever the user says "open Chrome and look for..."
    or "search Chrome for...". Figures out on its own whether the input is a URL
    or something to search.

    Args:
        query_or_url: A full URL (e.g. "youtube.com") OR a search phrase
            (e.g. "songs in the movie Karuppu").

    Returns:
        Confirmation message.
    """
    text = query_or_url.strip()
    looks_like_url = (
        text.startswith(("http://", "https://"))
        or (" " not in text and "." in text)  # e.g. "imdb.com"
    )
    if looks_like_url:
        url = text if text.startswith(("http://", "https://")) else "https://" + text
        ok, out = _open_in_chrome(url)
        return f"Opening {url} in Chrome." if ok else f"Could not open Chrome: {out}"

    url = "https://www.google.com/search?q=" + quote_plus(text)
    ok, out = _open_in_chrome(url)
    return f"Searching Chrome for '{text}'." if ok else f"Could not open Chrome: {out}"


def web_search(query: str) -> str:
    """Run a web search and show the results page in Chrome.

    Args:
        query: What to search for.

    Returns:
        Confirmation message.
    """
    url = "https://www.google.com/search?q=" + quote_plus(query)
    ok, out = _open_in_chrome(url)
    if ok:
        return f"Searching for '{query}'."
    # Last resort: the generic system search intent.
    ok, out = _am(["-a", "android.intent.action.WEB_SEARCH", "--es", "query", query])
    return f"Searching for '{query}'." if ok else f"Could not search: {out}"


def set_alarm(hour: int, minute: int, label: str = "") -> str:
    """Set an alarm on the phone's clock.

    Args:
        hour: Hour in 24-hour format (0-23).
        minute: Minute (0-59).
        label: Optional name for the alarm.

    Returns:
        Confirmation message.
    """
    args = ["-a", "android.intent.action.SET_ALARM",
            "--ei", "android.intent.extra.alarm.HOUR", str(hour),
            "--ei", "android.intent.extra.alarm.MINUTES", str(minute),
            "--ez", "android.intent.extra.alarm.SKIP_UI", "true"]
    if label:
        args += ["--es", "android.intent.extra.alarm.MESSAGE", label]
    ok, out = _am(args)
    when = f"{hour:02d}:{minute:02d}"
    return f"Alarm set for {when}." if ok else f"Could not set alarm: {out}"


def set_timer(seconds: int, label: str = "") -> str:
    """Start a countdown timer on the phone's clock.

    Args:
        seconds: Timer length in seconds.
        label: Optional name for the timer.

    Returns:
        Confirmation message.
    """
    args = ["-a", "android.intent.action.SET_TIMER",
            "--ei", "android.intent.extra.alarm.LENGTH", str(seconds),
            "--ez", "android.intent.extra.alarm.SKIP_UI", "true"]
    if label:
        args += ["--es", "android.intent.extra.alarm.MESSAGE", label]
    ok, out = _am(args)
    mins = seconds // 60
    secs = seconds % 60
    pretty = f"{mins}m {secs}s" if mins else f"{secs}s"
    return f"Timer started for {pretty}." if ok else f"Could not set timer: {out}"


def start_call(number: str) -> str:
    """Place a phone call to a number directly and hands-free - it ACTUALLY
    starts the call, the user does not tap anything. Use this whenever the user
    says "call <someone>". Requires the Termux:API app with the Phone/Call
    permission granted. If the direct call can't go through (permission missing),
    it falls back to opening the dialer with the number ready to tap.

    Args:
        number: The phone number to dial.

    Returns:
        Confirmation message.
    """
    # Primary: Termux:API places the call for real (needs Phone permission).
    ok, out = _run(["termux-telephony-call", number], timeout=15)
    low = (out or "").lower()
    if ok and "error" not in low and "permission" not in low and "denied" not in low:
        return f"Calling {number} now, sir."
    # Fallback: open the dialer pre-filled so he can tap the call button.
    ok2, out2 = _am(["-a", "android.intent.action.DIAL", "-d", f"tel:{number}"])
    if ok2:
        return (f"I've put {number} in the dialer, sir - tap call. (For fully "
                f"hands-free calling, grant the Termux:API app the Phone "
                f"permission in Android settings.)")
    return f"I couldn't start the call: {out or out2}"


# Cache the raw contacts list so we don't re-query Termux:API on every lookup
# and every voice transcription. Refreshed lazily every few minutes.
_contacts_cache = {"data": None, "at": 0.0}


def _get_contacts(max_age=180):
    """Return the phone's contacts as a list of dicts, cached briefly. On any
    failure returns an empty list (callers handle that)."""
    now = time.monotonic()
    if _contacts_cache["data"] is not None and now - _contacts_cache["at"] < max_age:
        return _contacts_cache["data"]
    ok, out = _run(["termux-contact-list"], timeout=10)
    data = []
    if ok:
        try:
            data = json.loads(out)
        except Exception:
            data = []
    _contacts_cache["data"] = data
    _contacts_cache["at"] = now
    return data


def find_contact(name: str) -> str:
    """Look up a phone number from the phone's contacts by name. ALWAYS use this
    FIRST when the user names a person (e.g. "text Selva", "call Amma", "message
    Ravi on WhatsApp") instead of giving a number - take the number it returns
    and pass it to send_sms, compose_sms, send_whatsapp, or start_call.
    Matching is fuzzy, so slightly misheard names (e.g. "silver" for "Selva")
    still match. Requires the Termux:API app and Contacts permission.

    Args:
        name: The contact's name, or part of it (case-insensitive).

    Returns:
        Matching contact name(s) and number(s), or a note if none found.
    """
    contacts = _get_contacts()
    if not contacts:
        return ("I couldn't read your contacts. Is the Termux:API app installed "
                "with Contacts permission granted, sir?")

    q = name.strip().lower()

    # Pass 1: exact substring match (handles "alex" -> "Alex Example").
    exact, seen = [], set()
    for c in contacts:
        cname = (c.get("name") or "")
        cnum = (c.get("number") or "")
        if q and q in cname.lower():
            key = (cname, cnum)
            if key not in seen:
                seen.add(key)
                exact.append(f"{cname}: {cnum}")
    if exact:
        return _fmt_matches(exact)

    # Pass 2: fuzzy match on each word of each contact name, for mis-hearings.
    scored = []
    for c in contacts:
        cname = (c.get("name") or "")
        cnum = (c.get("number") or "")
        best = 0.0
        for word in cname.lower().split():
            best = max(best, difflib.SequenceMatcher(None, q, word).ratio())
        # Also score against the whole name for multi-word queries.
        best = max(best, difflib.SequenceMatcher(None, q, cname.lower()).ratio())
        if best >= 0.6:
            scored.append((best, f"{cname}: {cnum}"))
    scored.sort(reverse=True)
    fuzzy, seen = [], set()
    for _, s in scored:
        if s not in seen:
            seen.add(s)
            fuzzy.append(s)
    if fuzzy:
        return _fmt_matches(fuzzy[:5])

    return (f"I couldn't find anyone like '{name}' in your contacts, sir. "
            f"Could you say the name again, or give me the number?")


def _fmt_matches(matches):
    if len(matches) == 1:
        return "Found - " + matches[0]
    if len(matches) > 8:
        return "Several matches: " + "; ".join(matches[:8]) + " (and more)."
    return "Found these - " + "; ".join(matches)


def compose_sms(number: str, message: str) -> str:
    """Open the messaging app pre-filled with a text, ready for you to send.
    This does NOT send automatically - you tap send yourself.

    Args:
        number: Recipient phone number.
        message: The message text.

    Returns:
        Confirmation message.
    """
    ok, out = _am(["-a", "android.intent.action.SENDTO",
                   "-d", f"sms:{number}", "--es", "sms_body", message])
    return f"Message to {number} is ready. Tap send when you're happy with it." \
        if ok else f"Could not open messages: {out}"


def send_sms(number: str, message: str) -> str:
    """Send an SMS text message IMMEDIATELY and hands-free - no tap needed.
    Use this when the user wants you to actually send a text. Requires the
    Termux:API app installed and SMS permission granted. If the user wants to
    review the text before sending, use compose_sms instead.

    Args:
        number: Recipient phone number.
        message: The message text to send.

    Returns:
        Confirmation message.
    """
    ok, out = _run(["termux-sms-send", "-n", number, message], timeout=15)
    if ok:
        return f"Message sent to {number}, sir."
    return (f"I couldn't send it: {out}. Make sure the Termux:API app is "
            f"installed and SMS permission is granted.")


def send_whatsapp(number: str, message: str = "") -> str:
    """Open a WhatsApp chat with a contact, message typed in and ready to send.
    On this phone you tap the send arrow yourself (fully automatic sending needs
    extra device setup). Use for "message X on WhatsApp".

    Args:
        number: Recipient's number in international format (country code + number,
            e.g. "919876543210"). Plus signs and spaces are fine, they're stripped.
        message: Optional text to pre-fill in the chat box.

    Returns:
        Confirmation message.
    """
    digits = "".join(ch for ch in number if ch.isdigit())
    url = f"https://wa.me/{digits}"
    if message:
        url += "?text=" + quote_plus(message)
    ok, out = _am(["-a", "android.intent.action.VIEW", "-d", url])
    if ok:
        return (f"WhatsApp is open with your message to {number} ready - "
                f"tap the send arrow, sir.")
    return f"Could not open WhatsApp: {out}"


def open_instagram(action: str = "", target: str = "") -> str:
    """Open Instagram, optionally jumping straight to a section.

    Args:
        action: What to open. One of:
            "" / "home"        -> just open Instagram,
            "profile"          -> open a user's profile (set `target` = username),
            "messages" / "dms" -> open the Direct message inbox,
            "search"           -> search Instagram (set `target` = search text).
        target: The username to open, or the text to search, depending on action.

    Returns:
        Confirmation message.
    """
    a = action.strip().lower()
    if a == "profile" and target:
        u = target.lstrip("@")
        ok, _ = _open_first([f"instagram://user?username={u}",
                             f"https://www.instagram.com/{u}/"])
        return f"Opening {target}'s Instagram profile." if ok \
            else "I couldn't open that Instagram profile."
    if a in ("messages", "dms", "dm", "inbox"):
        ok, _ = _open_first(["instagram://direct-inbox",
                             "https://www.instagram.com/direct/inbox/"])
        return "Opening your Instagram messages, sir." if ok \
            else "I couldn't open Instagram messages."
    if a == "search" and target:
        ok, _ = _am(["-a", "android.intent.action.VIEW", "-d",
                     "https://www.instagram.com/explore/search/keyword/?q="
                     + quote_plus(target)])
        return f"Searching Instagram for '{target}'." if ok \
            else "I couldn't search Instagram."
    # Default: just open the app.
    if _launch_package("com.instagram.android"):
        return "Opening Instagram."
    ok, _ = _am(["-a", "android.intent.action.VIEW",
                 "-d", "https://www.instagram.com/"])
    return "Opening Instagram." if ok else "I couldn't open Instagram."


# --- Knowledge, messages, memory, reminders ------------------------------

def answer_from_web(question: str) -> str:
    """Look something up on the web and return a short answer. Use ONLY for
    questions that need CURRENT or real-time facts: today's weather, latest news,
    live cricket/football scores, current prices, "is <place> open now", or very
    recent events. For general knowledge you already know (movies, songs, history,
    science, etc.), answer directly instead - do NOT use this tool for those.

    Args:
        question: The question to look up.

    Returns:
        A short spoken-friendly answer.
    """
    ask_text = (question + "\n\nAnswer in 1-3 short sentences, plainly, suitable "
                "for reading aloud.")
    # Preferred: grounded with live web search.
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=ask_text,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        text = (resp.text or "").strip()
        if text:
            return text
    except Exception:
        pass
    # Fallback: plain answer from the model's own knowledge (still useful if the
    # grounding tool isn't available on this SDK/model).
    try:
        resp = client.models.generate_content(model=MODEL, contents=ask_text)
        return (resp.text or "").strip() or "I couldn't find a clear answer, sir."
    except Exception as e:
        return f"I couldn't look that up just now, sir: {str(e)[:120]}"


def read_messages(count: int = 5, from_contact: str = "") -> str:
    """Read the most recent RECEIVED SMS text messages. Use for "read my
    messages", "any new messages", "what did <name> text me". Requires Termux:API
    with SMS permission.

    Args:
        count: How many recent inbox messages to read (default 5, max 15).
        from_contact: Optional name/number to filter by (e.g. "Selva").

    Returns:
        The messages as readable text, or a note if there are none.
    """
    n = max(1, min(int(count) if str(count).isdigit() else 5, 15))
    ok, out = _run(["termux-sms-list", "-t", "inbox", "-l", str(n)], timeout=12)
    if not ok:
        return ("I couldn't read your messages, sir. Is the Termux:API app "
                "installed with SMS permission granted?")
    try:
        msgs = json.loads(out)
    except Exception:
        return "I couldn't read your messages, sir."
    if from_contact:
        f = from_contact.strip().lower()
        msgs = [m for m in msgs
                if f in (m.get("sender") or "").lower()
                or f in (m.get("number") or "").lower()]
    if not msgs:
        return "You have no messages matching that, sir."
    lines = []
    for m in msgs[:n]:
        who = m.get("sender") or m.get("number") or "Unknown"
        body = (m.get("body") or "").strip()
        lines.append(f"From {who}: {body}")
    return " ... ".join(lines)


# Long-term memory: one fact per line in a plain text file, loaded into the
# persona at startup so JARVIS still knows it after a restart.
MEMORY_FILE = os.path.expanduser("~/.jarvis_memory.txt")


def _load_memories():
    try:
        with open(MEMORY_FILE) as f:
            return [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        return []


def remember(fact: str) -> str:
    """Save a fact to long-term memory so you still know it after a restart. Use
    when the user says "remember that ...", or shares a lasting preference, name,
    birthday, or detail worth keeping.

    Args:
        fact: The thing to remember, written as a short clear sentence.

    Returns:
        Confirmation message.
    """
    fact = " ".join(fact.strip().split())
    if not fact:
        return "There was nothing to remember, sir."
    with open(MEMORY_FILE, "a") as f:
        f.write(fact + "\n")
    return "I'll remember that, sir."


def list_memories() -> str:
    """List everything currently saved in your long-term memory.

    Returns:
        The saved facts, or a note if memory is empty.
    """
    mems = _load_memories()
    if not mems:
        return "I haven't saved anything to memory yet, sir."
    return "Here's what I remember - " + "; ".join(mems)


def forget(text: str) -> str:
    """Remove saved memories that mention the given text.

    Args:
        text: A word/phrase; any remembered fact containing it is removed.

    Returns:
        Confirmation message.
    """
    mems = _load_memories()
    keep = [m for m in mems if text.strip().lower() not in m.lower()]
    with open(MEMORY_FILE, "w") as f:
        f.write(("\n".join(keep) + "\n") if keep else "")
    removed = len(mems) - len(keep)
    return (f"Removed {removed} memory item(s), sir." if removed
            else "I didn't find that in memory, sir.")


def _fire_reminder(message):
    _run(["termux-notification", "--title", "JARVIS Reminder",
          "--content", message])
    subprocess.run(["termux-tts-speak", "-r", "0.95",
                    f"Reminder, sir. {message}"], capture_output=True, text=True)


def set_reminder(minutes: float, message: str) -> str:
    """Set a reminder that notifies you and speaks after some minutes. Works while
    JARVIS is running. If it must wake you even when JARVIS is closed, use
    set_alarm (the phone clock) instead.

    Args:
        minutes: How many minutes from now to remind (can be fractional).
        message: What to remind about.

    Returns:
        Confirmation message.
    """
    try:
        secs = max(1.0, float(minutes) * 60.0)
    except (TypeError, ValueError):
        return "How many minutes should I wait, sir?"
    t = threading.Timer(secs, _fire_reminder, args=(message,))
    t.daemon = True
    t.start()
    mins = secs / 60.0
    when = f"{mins:.0f} minutes" if mins >= 1 else f"{secs:.0f} seconds"
    return f"I'll remind you in {when}, sir: {message}."


# Notes: each note is {title, content, date, time}, stored as JSON so they can be
# searched by title, keyword, name, or date. Separate from long-term `remember`.
# NOTE: this is JARVIS's OWN note store - it CANNOT read the phone's Notes app.
NOTES_FILE = os.path.expanduser("~/.jarvis_notes.json")


def _load_notes():
    try:
        with open(NOTES_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, ValueError):
        return []


def _save_notes(notes):
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, ensure_ascii=False, indent=1)


def take_note(content: str, title: str = "") -> str:
    """Save a note, detail, or to-do so it can be found later. Use for "note down
    ...", "save my laptop password ...", "selva's birthday is ...", "shopping
    list: ...", "note selva's room address ...". ALWAYS give a short, clear title
    or category so it's easy to find later (e.g. "Shopping", "Laptop password",
    "Selva birthday", "Selva address"). Each note is saved with the date and time.

    Args:
        content: The note text / detail to save.
        title: A short title or category for easy lookup (strongly recommended).

    Returns:
        Confirmation message.
    """
    content = " ".join(content.strip().split())
    if not content:
        return "What should I note down, sir?"
    now = datetime.now()
    note = {
        "title": " ".join(title.strip().split()),
        "content": content,
        "date": now.strftime("%d/%m/%Y"),
        "time": now.strftime("%I:%M %p"),
    }
    notes = _load_notes()
    notes.append(note)
    _save_notes(notes)
    # Read back the full details so JARVIS can confirm exactly what was stored.
    return (f"Saved, sir. Title: {note['title'] or '(none)'}. "
            f"Date: {note['date']} at {note['time']}. Content: {content}.")


def read_notes(query: str = "") -> str:
    """Read saved notes. With a query, returns only the notes that match (search
    by title, content, name, or date). Without a query, lists ALL notes in order.
    Use the query to answer specific asks like "my laptop password", "selva's
    birthday", "shopping list", "selva room address", or "notes from 09/07/2026".
    For "the first note" / "the last note", read all and pick the first/last.

    Args:
        query: Optional search text - a title, keyword, name, or a date like
            "09/07/2026". Leave empty to list everything.

    Returns:
        The matching notes with their title, date and content.
    """
    notes = _load_notes()
    if not notes:
        return "You have no notes saved, sir."
    q = query.strip().lower()
    if q:
        qwords = [w for w in q.split() if len(w) >= 2]
        need = max(1, (len(qwords) + 1) // 2)   # at least half the words must hit
        scored = []
        for n in notes:
            blob = f"{n.get('title','')} {n.get('content','')} {n.get('date','')}".lower()
            if q in blob:                        # whole phrase present -> best
                scored.append((1000, n))
                continue
            bwords = blob.split()
            hit = sum(1 for qw in qwords if qw in blob or
                      any(difflib.SequenceMatcher(None, qw, bw).ratio() >= 0.82
                          for bw in bwords))
            if hit >= need:
                scored.append((hit, n))
        if not scored:
            return f"I couldn't find any note about '{query}', sir."
        scored.sort(key=lambda s: s[0], reverse=True)
        notes = [n for _, n in scored]
    lines = []
    for i, n in enumerate(notes, 1):
        t = f"{n['title']} - " if n.get("title") else ""
        lines.append(f"{i}. ({n.get('date','')}) {t}{n.get('content','')}")
    return "\n".join(lines)


def delete_note(query: str) -> str:
    """Delete saved note(s) matching the given text (title or content).

    Args:
        query: Text identifying which note(s) to delete.

    Returns:
        Confirmation message.
    """
    notes = _load_notes()
    q = query.strip().lower()
    keep = [n for n in notes
            if q not in f"{n.get('title','')} {n.get('content','')}".lower()]
    removed = len(notes) - len(keep)
    _save_notes(keep)
    return (f"Deleted {removed} note(s), sir." if removed
            else f"I couldn't find a note about '{query}', sir.")


def clear_notes() -> str:
    """Delete ALL saved notes.

    Returns:
        Confirmation message.
    """
    try:
        os.remove(NOTES_FILE)
    except FileNotFoundError:
        pass
    return "All notes cleared, sir."


ALL_TOOLS = [
    set_flashlight, get_battery_status, get_volume, set_volume,
    set_brightness, set_auto_brightness, vibrate, get_time_and_date,
    get_clipboard, set_clipboard, show_notification,
    open_app, open_url, open_in_chrome, web_search,
    set_alarm, set_timer, start_call, find_contact,
    compose_sms, send_sms, send_whatsapp, open_instagram,
    answer_from_web, read_messages,
    remember, list_memories, forget, set_reminder,
    take_note, read_notes, delete_note, clear_notes,
]


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------

SYSTEM = f"""You are JARVIS, a personal AI assistant modeled after Tony Stark's JARVIS from Iron Man.
You belong to {OWNER}. You address him by name occasionally, or as 'sir'.
Your tone: British butler precision meets close friend - warm, dryly witty, never sycophantic.
You are his confidant as well as his assistant.

How to think and act:
- ACCURACY MATTERS MORE THAN SPEED. {OWNER} trusts your answers and often cannot
  tell if you are wrong - a wrong fact will mislead him. So for ANY factual
  question about the real world, USE the answer_from_web tool to look it up
  instead of answering from memory: which movie/album a song is from, who did
  something, dates, names, places, lyrics, statistics, sports, "is it true
  that...", current events, weather, scores, prices. Do NOT answer these from
  memory, and NEVER guess a name/title/date - your memory hallucinates on niche
  facts (especially Tamil cinema/songs). Look it up.
- You MAY answer directly, without a tool, ONLY for: casual conversation,
  opinions and advice, jokes, explaining general concepts, maths, and
  language/translation help. Never say "I don't have access" - if it's a fact,
  look it up; if it's chat, just answer.
- If even a web lookup is not clearly certain, say you are not fully sure rather
  than stating a guess as fact.
- Be decisive and capable, like a sharp personal assistant. When a request is
  clear, DO IT with the tools - don't ask for permission on simple, safe actions.
- Reason step by step. A single request may need SEVERAL tools in sequence
  (e.g. open an app, then act in it). Chain them to finish the whole job. For a
  task like "open Chrome and search karuppu movie songs", do the whole thing:
  open + search in one go.
- Know your limits honestly. Your tools can LAUNCH apps, open URLs, run web
  searches, send SMS, prepare WhatsApp messages, set alarms, and control device
  settings. You CANNOT see the screen or tap buttons inside a web page or app -
  so you cannot click a "download" button, pick a result, or navigate a website.
  When a request needs that (e.g. "download the song"), do the part you can
  (open the search) and tell the user plainly you can't tap the download for
  them yet - don't pretend you did.
- Infer sensible defaults instead of over-asking. If the user says "turn off"
  right after talking about the flashlight, they mean the flashlight - just do
  it. Only ask a clarifying question when a request is genuinely ambiguous or
  could do something harmful/irreversible.
- Use conversation context: track what was just discussed and resolve pronouns
  ("it", "that", "him") from it.
- {OWNER} is not a fluent English speaker. He may speak or type in casual, broken
  English or in "Tanglish" (Tamil mixed with English, written in English letters,
  e.g. "instagram open pannu", "selva ku message anupu"). Understand his INTENT
  generously - never nitpick grammar or wording, and use context to fill gaps.
  Reply in simple, clear language; a little Tanglish back is fine if it helps.
- Voice commands are transcribed and may have small errors. If a word looks a bit
  off but the intent is obvious, act on the intent instead of asking him to repeat.

Rules:
- When the user asks for something on the phone, use the available tools - don't
  just describe what you would do.
- When the user names a PERSON instead of a number (text/call/WhatsApp someone),
  call find_contact FIRST to get their number, then do the action. Only ask the
  user for a number if find_contact finds no match. If several numbers match,
  briefly ask which one.
- REMEMBER the person across the whole conversation, like a real assistant. Once
  {OWNER} has picked or confirmed a specific contact (e.g. he chose "Selva
  Acme" among several "Selva" matches, or you just messaged/called that
  person), then if he later refers to the same person by a shorter or partial
  name ("call Selva", "message him again", "him"), REUSE that same contact and
  number from earlier in this conversation. Do NOT call find_contact again and do
  NOT re-ask "which Selva" - you already know which one he means. Only look up or
  ask again if he clearly names a DIFFERENT person, or explicitly says he means
  another one.
- CHOOSING HOW TO SEND A MESSAGE - follow this strictly:
  * If the user just says "message", "text", "sms", "send", or "anupu"/"pannu"
    WITHOUT naming an app -> use send_sms (normal SMS). This is the default.
  * Only use send_whatsapp when they actually say "WhatsApp".
  * Only use open_instagram for Instagram when they say "Instagram".
  * Never send a plain "message" through WhatsApp or any app unless they named it.
- SAVING NOTES - handle titles carefully:
  * ALWAYS set a short, clear title yourself from what the note is about, even if
    {OWNER} did not say a title. E.g. "this is selva's address, save it" -> title
    "Selva address"; "save selva's laptop password abc" -> title "Selva laptop
    password". Match it to the person and the kind of detail.
  * If the title {OWNER} gives seems to MISMATCH the content - e.g. he says save a
    phone NUMBER but calls it an "address", or saves an address but calls it a
    "number" - do NOT just save and do NOT refuse. Point out the mismatch and ask
    him to confirm first (e.g. "That looks like a phone number, not an address -
    still save it as 'Selva address', sir?"). Save only after he confirms.
  * After saving, tell him the saved note's title, date, time and content back.
- After running a tool, briefly confirm what you did - don't stay silent.
- For calls and texts, some tools only PREPARE the action - the user taps
  send/call. Say so clearly when that's the case.
- Be concise - your replies are spoken aloud, so keep sentences short and natural.
- If you genuinely can't do something, say so plainly and, if useful, suggest an
  alternative.
- Open with a brief greeting only on the very first turn.
- If the user's words are clearly garbled (speech recognition is imperfect) AND
  you can't reasonably guess the intent, ask them to repeat. Otherwise, proceed.
"""

def _new_chat(model, history=None):
    """Create a chat session on `model`, optionally seeded with prior history.
    Long-term memories are loaded fresh here so they're always in the persona,
    even after a restart or a mid-session model fallback."""
    sys_text = SYSTEM
    mems = _load_memories()
    if mems:
        sys_text += ("\n\nLong-term memory - things you remember about " + OWNER
                     + " from earlier sessions (use them naturally):\n- "
                     + "\n- ".join(mems))
    # Try with the tool-call cap; if this SDK version dislikes that config, fall
    # back to a plain config so the assistant always starts.
    try:
        cfg = types.GenerateContentConfig(
            system_instruction=sys_text,
            tools=ALL_TOOLS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=6,   # a bug can't loop tools forever
            ),
        )
        return client.chats.create(model=model, config=cfg, history=history or [])
    except Exception:
        cfg = types.GenerateContentConfig(system_instruction=sys_text, tools=ALL_TOOLS)
        return client.chats.create(model=model, config=cfg, history=history or [])


active_model = MODEL
chat = _new_chat(active_model)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def speak(text):
    print(f"JARVIS: {text}\n")
    subprocess.run(["termux-tts-speak", "-r", "0.95", text])


def _stop_tts():
    """Force-stop any speech that's still playing (used when interrupted)."""
    subprocess.run(["pkill", "-f", "termux-tts-speak"],
                   capture_output=True, text=True)


def _flush_input():
    """Discard any keystrokes typed while JARVIS was busy. Without this, Enters
    you pressed while it was talking/thinking get buffered and then auto-trigger
    the mic when we return to the prompt - which felt like it 'looped on its own'."""
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


# Where a recorded voice command is stashed before transcribing.
VOICE_FILE = os.path.expanduser("~/.jarvis_voice.m4a")


# Model used for transcription. flash is strong at multilingual audio; kept
# separate from the chat model so a chat fallback doesn't weaken transcription.
STT_MODEL = os.getenv("GEMINI_STT_MODEL", "gemini-2.5-flash")


def transcribe(path):
    """Send a recorded audio command to Gemini and get back the text.

    IMPORTANT lesson learned: do NOT feed contact names, app names, or example
    commands into this prompt as 'hints'. Doing so PRIMES the model to hallucinate
    those exact words onto the end of the real speech (it was gluing a phantom
    'selva ... message pannu' onto unrelated commands and onto trailing silence).
    Faithful, un-primed transcription + temperature 0 is the correct approach; the
    fuzzy find_contact() handles any misheard names later, at lookup time.
    Returns '' when there is no clear speech."""
    prompt = (
        "Transcribe this short audio of a voice command to a phone assistant. "
        "The speaker is Indian and may mix casual English with Tamil words. Write "
        "any Tamil words in ENGLISH letters (Latin/Tanglish), not Tamil script.\n"
        "Rules you MUST follow:\n"
        "1. Write ONLY the words that are actually, clearly spoken.\n"
        "2. Do NOT invent, guess, autocomplete, or add any command, name, or "
        "phrase that is not clearly heard. Do NOT repeat phrases.\n"
        "3. Ignore silence and background noise - do not transcribe them.\n"
        "4. If there is no clear human speech at all, reply with exactly: NOSPEECH\n"
        "Output ONLY the transcript, nothing else."
    )
    # Inline audio = one round-trip (faster). temperature 0 = no 'creative' filling.
    try:
        with open(path, "rb") as f:
            audio = f.read()
        mime = "audio/mp4" if path.endswith(".m4a") else "audio/aac"
        resp = client.models.generate_content(
            model=STT_MODEL,
            contents=[types.Part.from_bytes(data=audio, mime_type=mime), prompt],
            config=types.GenerateContentConfig(temperature=0),
        )
        text = (resp.text or "").strip()
        return "" if "NOSPEECH" in text.upper() else text
    except Exception as e:
        print(f"(voice transcription failed: {str(e)[:150]})")
        return ""


def _mic_start():
    """Start the background recorder. Returns True on success. Tries AAC first,
    falls back to the default encoder if this Termux:API build rejects -e."""
    for args in (["-e", "aac"], []):
        r = subprocess.run(
            ["termux-microphone-record", "-f", VOICE_FILE, "-l", "120"] + args,
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return True
    return False


# How many seconds to record before auto-stopping. You can change this any time
# by setting JARVIS_LISTEN_SECS in your .env (e.g. 12 for longer commands).
LISTEN_SECS = float(os.getenv("JARVIS_LISTEN_SECS", "9"))


def _wait_for_stop(max_secs):
    """Block until EITHER the user presses Enter OR max_secs elapse - whichever
    comes first. Uses select() on stdin, so no second key-press is required: if
    the user just stops talking and waits, recording ends on its own."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], max_secs)
        if ready:
            sys.stdin.readline()   # consume the Enter they pressed to finish early
    except Exception:
        time.sleep(max_secs)       # if select isn't available, just wait it out


def listen():
    """Record a voice command and transcribe it with Gemini.

    Key design points from real-world use:
    - The mic needs a moment to warm up; we wait + buzz BEFORE you speak so your
      first word isn't clipped ("jarvis send..." was getting lost).
    - You press Enter ONCE to start. Recording then auto-stops after LISTEN_SECS
      seconds so you don't have to press anything again - just stop talking. You
      CAN press Enter to finish sooner if you're done early.
    - If nothing was heard, we say so and let you retry - we do NOT fall back to
      the phone's weak recognizer, which was mishearing and truncating."""
    # Clear any previous session/file.
    subprocess.run(["termux-microphone-record", "-q"], capture_output=True, text=True)
    try:
        if os.path.exists(VOICE_FILE):
            os.remove(VOICE_FILE)
    except OSError:
        pass

    # Audible "go" cue BEFORE the mic starts, so JARVIS's own voice isn't recorded
    # (vibrate doesn't work on this phone). By the time you react and start
    # speaking, the mic is warmed up, so your first word isn't clipped.
    subprocess.run(["termux-tts-speak", "-r", "1.1", "Yes sir?"],
                   capture_output=True, text=True)

    if not _mic_start():
        print("(couldn't start the mic - check Microphone permission for "
              "Termux:API. You can type your command instead.)")
        return ""

    time.sleep(0.6)   # mic warm-up
    print(f">>> Speak now. Auto-stops in {LISTEN_SECS:.0f}s, "
          f"or press Enter the moment you're done.")
    try:
        _wait_for_stop(LISTEN_SECS)
    finally:
        # Always stop the mic, even if interrupted with Ctrl+C, so it never keeps
        # recording in the background.
        subprocess.run(["termux-microphone-record", "-q"],
                       capture_output=True, text=True)
    time.sleep(0.3)   # let the file flush to disk

    if not os.path.exists(VOICE_FILE) or os.path.getsize(VOICE_FILE) < 1200:
        print("(I didn't catch anything - try again, or type it.)")
        return ""

    text = transcribe(VOICE_FILE)
    if text:
        print(f'(heard: "{text}")')   # you can see exactly what it understood
        return text
    print("(I couldn't make out what you said, sir - please say it again.)")
    return ""


def ask(user_text, max_retries=3, rounds=2, round_pause=5.0):
    """Send a message, retrying transient overloads and falling back across
    models. Makes `rounds` full sweeps over every fallback model (with a pause
    between sweeps) before giving up, so short overload spikes recover on their
    own. Returns the reply text, or None only if everything is exhausted."""
    global chat, active_model

    for round_no in range(rounds):
        if round_no > 0:
            # Every model was busy last sweep — wait a bit, then try them all
            # again from the top. Spikes usually clear within a few seconds.
            print(f"(all models busy, waiting {round_pause:.0f}s before retrying...)")
            time.sleep(round_pause)

        for model in FALLBACK_MODELS:
            if model != active_model:
                # Move to the next model but keep the whole conversation so
                # context isn't lost.
                print(f"({active_model} unavailable, switching to {model}...)")
                chat = _new_chat(model, chat.get_history())
                active_model = model

            delay = 1.0
            for attempt in range(max_retries):
                try:
                    resp = chat.send_message(user_text)
                    return resp.text
                except errors.APIError as e:
                    code = getattr(e, "code", None)
                    if code in RETRYABLE_CODES and attempt < max_retries - 1:
                        reason = "rate limit/quota" if code == 429 else "server busy"
                        print(f"({model}: {reason} [{code}], retrying in {delay:.0f}s...)")
                        time.sleep(delay)
                        delay = min(delay * 2, 8)   # 1s, 2s, 4s, 8s cap
                        continue
                    # Not retryable, or out of attempts on this model.
                    if code in RETRYABLE_CODES:
                        break               # try the next fallback model
                    print(f"(API error {code}: {str(e)[:200]})")
                    return None
                except Exception as e:
                    print(f"(unexpected error: {str(e)[:200]})")
                    return None

    return None


# Saying (or typing) any of these shuts JARVIS down.
SHUTDOWN_PHRASES = {
    "goodbye", "good bye", "goodnight", "good night", "bye", "night",
}


def _is_shutdown(text):
    """True if the user is telling JARVIS to shut down. Tolerant of punctuation,
    a 'jarvis' anywhere (e.g. 'Jarvis, good night.'), and small filler words, so
    both 'good night' and 'jarvis good night' work."""
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    words = [w for w in s.split() if w != "jarvis"]
    clean = " ".join(words)
    if clean in SHUTDOWN_PHRASES:
        return True
    # Lenient: a short utterance that contains a farewell phrase (e.g.
    # "ok good night sir", "jarvis good night da").
    if len(words) <= 4:
        for p in ("good night", "goodnight", "good bye", "goodbye"):
            if p in clean:
                return True
    return False


def get_input():
    raw = input(f"{OWNER} > ").strip()
    if raw == "":
        return listen()
    return raw


def main():
    speak(f"JARVIS online. Good to see you, {OWNER}.")
    print(f"\nModel: {MODEL}")
    print(f"Tools loaded: {len(ALL_TOOLS)}")
    print("Press Enter (empty) to SPEAK, or TYPE a message and press Enter.")
    print("Press Ctrl+C to INTERRUPT Jarvis (stop talking / stuck). Ctrl+C twice = quit.")
    print("Say or type 'goodbye' or 'good night' to shut down.\n")

    interrupts = 0   # consecutive Ctrl+C presses at the idle prompt
    while True:
        # Throw away any Enters/keys mashed while JARVIS was busy, so they don't
        # auto-fire the mic. This is the fix for "mic turns on by itself".
        _flush_input()
        try:
            user_input = get_input()
            interrupts = 0
        except KeyboardInterrupt:
            # Ctrl+C while waiting at the prompt. Once = ignore (just a break),
            # twice in a row = shut down.
            interrupts += 1
            if interrupts >= 2:
                print()
                speak("Shutting down. Goodbye, sir.")
                break
            print("\n(Interrupted. Press Enter to speak, type a command, or "
                  "Ctrl+C again to quit.)\n")
            continue
        except EOFError:
            print()
            speak("Shutting down. Goodbye, sir.")
            break

        if not user_input:
            print("(nothing heard, try again)\n")
            continue
        print(f"{OWNER} said: {user_input}\n")
        if _is_shutdown(user_input):
            speak("Goodnight, sir. Call me when you need me.")
            break

        # Ctrl+C during thinking or speaking cancels THIS command and drops back
        # to the prompt - it does not quit the whole assistant.
        try:
            reply = ask(user_input)
            if reply is None:
                speak("The servers are busy right now, sir. Give me a moment "
                      "and try again.")
            else:
                speak(reply)
        except KeyboardInterrupt:
            _stop_tts()
            print("\n(Stopped, sir. What next?)\n")
            continue


if __name__ == "__main__":
    main()
