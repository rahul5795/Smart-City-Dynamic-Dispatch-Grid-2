import hashlib
import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import math
from collections import Counter

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

try:
    from groq import Groq
except ImportError:
    Groq = None

try:                                    # free speech-to-text used when no Groq key is set
    import speech_recognition as sr
except ImportError:
    sr = None

st.set_page_config(
    page_title="Smart City Dynamic Dispatch Grid",
    page_icon="🚑",
    layout="wide",
)

st.title("🚨 Smart City Dynamic Dispatch Grid")

# ----------------------- Sahayak 112 citizen app -----------------------
# The full app is one HTML file. Put sahayak112.html next to app.py (same folder / same GitHub repo).
SAHAYAK_FILE = Path(__file__).parent / "sahayak112.html"


@st.cache_data(show_spinner=False)
def load_sahayak(path: str, modified: float) -> str | None:
    """Read the app once; the cache refreshes automatically when the file changes."""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None
st.caption("Synthetic emergency-call triage, incident prioritization, and vehicle dispatch demo")
st.warning(
    "Prototype only: this uses synthetic data and a simplified grid. "
    "Do not use it to make real emergency-response decisions."
)

# ----------------------- City and fleet -----------------------
GRID_SIZE = 20
LANDMARKS = {
    "Central Hospital": (3, 3),
    "City Hospital": (3, 3),
    "Railway Station": (15, 4),
    "Main Railway Station": (15, 4),
    "River Bridge": (10, 10),
    "Main Bridge": (10, 10),
    "Central Market": (6, 14),
    "City Market": (6, 14),
    "North School": (4, 17),
    "South School": (16, 15),
    "Fire Station": (2, 10),
    "Police Station": (17, 8),
    "Bus Terminal": (12, 3),
    "Petrol Pump": (8, 6),
    "Old Temple": (5, 5),
    "Shopping Mall": (13, 13),
    "River Bank": (10, 8),
    "Industrial Area": (18, 17),
    "Stadium": (11, 16),
    "Airport Road": (18, 3),
}

VEHICLES = [
    {"Vehicle ID": "AMB-01", "Type": "Ambulance", "Station": (2, 3)},
    {"Vehicle ID": "AMB-02", "Type": "Ambulance", "Station": (4, 12)},
    {"Vehicle ID": "AMB-03", "Type": "Ambulance", "Station": (14, 4)},
    {"Vehicle ID": "AMB-04", "Type": "Ambulance", "Station": (16, 15)},
    {"Vehicle ID": "AMB-05", "Type": "Ambulance", "Station": (8, 8)},
    {"Vehicle ID": "FIRE-01", "Type": "Fire Truck", "Station": (2, 10)},
    {"Vehicle ID": "FIRE-02", "Type": "Fire Truck", "Station": (17, 8)},
    {"Vehicle ID": "FIRE-03", "Type": "Fire Truck", "Station": (11, 16)},
    {"Vehicle ID": "RES-01", "Type": "Rescue Team", "Station": (10, 10)},
    {"Vehicle ID": "RES-02", "Type": "Rescue Team", "Station": (18, 17)},
]

def make_city_graph(seed: int = 7) -> nx.Graph:
    rng = random.Random(seed)
    graph = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)
    for u, v in graph.edges:
        graph.edges[u, v]["minutes"] = rng.randint(1, 3)
        graph.edges[u, v]["closed"] = False
    # River-like barrier with three crossing points, matching the demo's project concept.
    # Remove horizontal crossings between x=9 and x=10 except at bridge rows.
    bridge_rows = {3, 10, 16}
    for y in range(GRID_SIZE):
        if y not in bridge_rows:
            a, b = (9, y), (10, y)
            if graph.has_edge(a, b):
                graph.remove_edge(a, b)
    return graph

def add_closures(graph: nx.Graph, closure_rows: list[int], closure_cols: list[int]) -> nx.Graph:
    graph = graph.copy()
    # Close selected horizontal/vertical street segments, not entire junctions.
    for y in closure_rows:
        for x in range(GRID_SIZE - 1):
            if graph.has_edge((x, y), (x + 1, y)):
                graph.remove_edge((x, y), (x + 1, y))
    for x in closure_cols:
        for y in range(GRID_SIZE - 1):
            if graph.has_edge((x, y), (x, y + 1)):
                graph.remove_edge((x, y), (x, y + 1))
    return graph

def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The AI response did not contain a JSON object.")
    return json.loads(text[start:end + 1])

# Severity guide from the project plan (section 3.2). When a call fits two levels, the higher one wins.
SEVERITY_GUIDE = {
    5: "Life in danger now",
    4: "Serious harm or spreading danger",
    3: "Injury, stable",
    2: "Damage, nobody hurt",
    1: "Information only",
}
SEVERITY_RULES = [   # (level, phrases, reason shown to the dispatcher)
    (5, ["trapped", "not breathing", "unconscious", "not moving", "heart attack", "drowning", "buried"], "someone's life is in danger now"),
    (4, ["spreading", "getting worse", "explosion", "severe bleeding", "heavy bleeding", "bleeding badly", "lot of blood",
         "people inside", "flames", "serious", "critical"], "serious harm or a danger that is spreading"),
    (3, ["injured", "hurt", "broken", "bleeding", "crash", "accident", "fell"], "someone is injured but stable"),
    (2, ["crack", "damage", " zznoharmzz "], "damage, nobody hurt"),
    (1, ["road blocked", "power cut", "power is gone", "blocked"], "information only"),
]
TYPE_RULES = [       # first match wins; the order mirrors how dispatch would respond
    ("Fire", ["fire", "smoke", "flames", "burning", "explosion", "gas leak", "gas smell"]),
    ("Rescue", ["trapped", "collapse", "building fell", "came down", "rubble", "flood", "drowning", "water rising", "crack", "wall"]),
    ("Medical", ["unconscious", "not breathing", "heart attack", "bleeding", "injured", "accident", "crash", "hurt",
                 "fell", "fainted", "broken"]),
]
DEFAULT_SEVERITY = {"Fire": 3, "Rescue": 4, "Medical": 3}
NEGATIONS = ["no one is hurt", "no one hurt", "nobody is hurt", "nobody hurt", "nobody is injured", "no injuries",
             "no one is injured", "not injured", "not hurt"]
PRANK_PHRASES = ["prank", "fake call", "nothing happened", "test call", "no emergency", "just kidding", "wrong number"]
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "several": 3, "many": 5}


def count_people(t: str) -> int | None:
    nums = [int(n) for n in re.findall(r"\b(\d{1,2})\b", t)]
    nums += [v for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", t)]
    return max(nums) if nums else None


def rule_based_triage(call_text: str) -> dict[str, Any]:
    """Fallback triage if Groq is not configured or its request fails.

    Uses the plan's severity guide and explains every level it picks. Confidence is not
    measured by rules, so it is reported as None instead of a made-up number.
    """
    t = " " + call_text.lower() + " "
    for neg in NEGATIONS:                                     # "nobody hurt" must not count as "hurt"
        t = t.replace(neg, " zznoharmzz ")
    location = match_location(call_text)
    if any(w in t for w in PRANK_PHRASES):
        return {"is_emergency": False, "incident_type": "Noise / prank", "severity": 1, "location": location,
                "summary": "Looks like a prank or non-emergency call; no vehicle is needed.",
                "severity_reason": "the caller says there is no real emergency", "people": None, "confidence": None}
    kind = next((k for k, words in TYPE_RULES if any(w in t for w in words)), None)
    keyword_type = kind is not None
    kind = kind or "Medical"
    hits = [(lvl, reason, next(w for w in words if w in t)) for lvl, words, reason in SEVERITY_RULES if any(w in t for w in words)]
    if hits:
        severity, reason, word = max(hits)                       # torn between levels: the higher one wins
        reason = f"the caller said “{word.strip() if 'zz' not in word else 'nobody hurt'}”"
    else:
        severity, reason = DEFAULT_SEVERITY[kind], "no details on harm, so the usual level for this type is used"
    people = count_people(t)
    if people and people >= 3 and severity == 4 and kind != "Fire":
        severity, reason = 5, reason + "; several people involved"
    where = location or "an unknown location"
    summary = f"{kind} reported near {where}" + (f", about {people} people involved" if people else "") + "."
    return {"is_emergency": True, "incident_type": kind, "severity": severity, "location": location, "type_from_keywords": keyword_type,
            "summary": summary, "severity_reason": reason, "people": people, "confidence": None}


def match_location(text: str) -> str | None:
    lower = text.lower()
    for name in sorted(LANDMARKS, key=len, reverse=True):
        if name.lower() in lower:
            return name
    aliases = {
        "hospital": "Central Hospital",
        "railway": "Railway Station",
        "bridge": "River Bridge",
        "market": "Central Market",
        "school": "North School",
        "mall": "Shopping Mall",
        "stadium": "Stadium",
        "airport": "Airport Road",
        "temple": "Old Temple",
        "petrol": "Petrol Pump",
        "industrial": "Industrial Area",
        "bus stand": "Bus Terminal",
    }
    for phrase, landmark in aliases.items():
        if phrase in lower:
            return landmark
    return None          # never invent a map point: the dispatcher calls back instead (plan, section 3.4)

def groq_triage(call_text: str, api_key: str, model: str) -> dict[str, Any]:
    if Groq is None:
        raise RuntimeError("Groq package is not installed. Install requirements.txt.")
    client = Groq(api_key=api_key)
    prompt = f"""
You are a cautious triage assistant for a SYNTHETIC emergency dispatch demonstration.
Extract a single call into valid JSON only with keys:
is_emergency (boolean), incident_type (one of Medical, Fire, Rescue, Noise / prank),
severity (integer 1-5, where 5 is most urgent), location (one exact string from allowed locations),
summary (short string), confidence (number 0-1).
Allowed locations: {list(LANDMARKS.keys())}.
Use the best matching location from the list. If no location is clear, use null.
Never invent a location or any other fact.
Severity guide (when torn between two levels, choose the higher one):
5 = life in danger now (trapped, not breathing, unconscious); 4 = serious harm or spreading danger (fire spreading,
heavy bleeding); 3 = injury, stable; 2 = damage, nobody hurt; 1 = information only.
Also return severity_reason (one short phrase quoting what in the call decides the level) and people (integer or null). Prank/test/non-emergency calls should have
is_emergency=false and incident_type="Noise / prank". Severity must be 1-5.
Call text: {call_text!r}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only a valid JSON object. This is a demo, not real emergency advice."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=350,
        response_format={"type": "json_object"},
    )
    data = parse_json_response(response.choices[0].message.content or "")
    data["severity"] = max(1, min(5, int(data.get("severity", 3))))
    if data.get("location") not in LANDMARKS:
        data["location"] = match_location(call_text)      # None if the call names no known place
        data["confidence"] = min(float(data.get("confidence") or 0.5), 0.4)
    data["is_emergency"] = bool(data.get("is_emergency", True))
    data["incident_type"] = str(data.get("incident_type", "Medical"))
    data["summary"] = str(data.get("summary", call_text[:180]))
    data["confidence"] = max(0.0, min(1.0, float(data.get("confidence") or 0.5)))
    data["severity_reason"] = str(data.get("severity_reason") or SEVERITY_GUIDE[data["severity"]].lower())
    try:
        data["people"] = int(data["people"]) if data.get("people") is not None else None
    except (TypeError, ValueError):
        data["people"] = None
    return data

def route_minutes(graph: nx.Graph, start: tuple[int, int], end: tuple[int, int]) -> float:
    # Weighted shortest path based on simulated travel minutes.
    return float(nx.shortest_path_length(graph, start, end, weight="minutes"))

def route_nodes(graph: nx.Graph, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    return nx.shortest_path(graph, start, end, weight="minutes")

def choose_vehicle(incident_type: str, location: str, graph: nx.Graph, available: list[dict]) -> tuple[dict | None, float | None]:
    required = {"Medical": "Ambulance", "Fire": "Fire Truck", "Rescue": "Rescue Team"}.get(incident_type)
    candidates = [v for v in available if v["Type"] == required]
    if not candidates:
        # Demo fallback: allow any vehicle if a matching type is unavailable.
        candidates = available
    if not candidates:
        return None, None
    if location not in LANDMARKS:
        return None, None                                   # no location yet: wait for the call-back
    target = LANDMARKS[location]
    scored = []
    for v in candidates:
        try:
            minutes = route_minutes(graph, v["Station"], target)
            scored.append((minutes, v))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    if not scored:
        return None, None
    minutes, vehicle = min(scored, key=lambda item: item[0])
    return vehicle, minutes

def draw_map(incidents: list[dict], assignments: list[dict], closures_rows: list[int], closures_cols: list[int]):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_title("Simulated city grid (not a real geographic map)")
    # Landmarks
    for name, (x, y) in LANDMARKS.items():
        ax.scatter(x, y, marker="s", s=35)
        ax.annotate(name, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    # Vehicle stations
    for v in VEHICLES:
        x, y = v["Station"]
        ax.scatter(x, y, marker="^", s=48)
    # Incidents
    for inc in incidents:
        if inc.get("is_emergency"):
            if inc.get("location") not in LANDMARKS:
                continue                                    # unknown location: not drawn on the map
            x, y = LANDMARKS[inc["location"]]
            ax.scatter(x, y, marker="X", s=90)
    for y in closures_rows:
        ax.axhline(y, linewidth=2, linestyle="--", alpha=0.55)
    for x in closures_cols:
        ax.axvline(x, linewidth=2, linestyle="--", alpha=0.55)
    ax.text(0.01, -0.13, "Squares = landmarks | Triangles = vehicle stations | X = emergency incidents | Dashed lines = simulated closures",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    return fig


SEV_COLOR = {5: "#D62839", 4: "#E46A1E", 3: "#D9A21B", 2: "#86AE3A", 1: "#8A97A5"}


def show_triage_card(rec: dict[str, Any]) -> None:
    """Dispatcher-friendly view of one triaged call; the raw JSON stays available for developers."""
    sev = int(rec.get("severity", 3))
    color = SEV_COLOR.get(sev, "#8A97A5")
    with st.container(border=True):
        top = st.columns([3, 2])
        top[0].markdown(f"### {rec['Call ID']}: {rec['incident_type']}")
        if str(rec.get("Source", "")).startswith("Voice"):
            top[0].caption(f"🎙️ Reported by voice, written by {rec.get('Voice engine') or 'speech recognition'}"
                           + (", then edited by hand" if "edited" in rec["Source"] else ""))
        method = rec.get("triage_method", "")
        top[1].markdown(
            f"<div style='text-align:right;margin-top:10px'><span style='background:{'#1F8A5B' if method == 'Groq AI' else '#1C5D99' if 'ML' in method else '#5B6B7A'};"
            f"color:#fff;border-radius:999px;padding:3px 10px;font-size:0.8rem;font-weight:600'>{method}</span></div>",
            unsafe_allow_html=True)
        if not rec.get("is_emergency"):
            st.info("Not an emergency: no vehicle will be sent. " + rec.get("summary", ""))
        c = st.columns(4)
        c[0].markdown(
            f"<div style='font-size:0.85rem;opacity:.75'>Severity</div>"
            f"<div style='display:inline-block;background:{color};color:#fff;border-radius:10px;padding:2px 12px;"
            f"font-size:1.6rem;font-weight:800'>{sev}</div> <span style='font-size:0.85rem'>of 5</span>",
            unsafe_allow_html=True)
        def fact(col, label, value, muted=False):
            col.markdown(f"<div style='font-size:0.85rem;opacity:.75'>{label}</div>"
                         f"<div style='font-size:1.25rem;font-weight:700;{'opacity:.6' if muted else ''}'>{value}</div>",
                         unsafe_allow_html=True)
        fact(c[1], "Location", rec.get("location") or "Unknown", muted=not rec.get("location"))
        fact(c[2], "People", rec.get("people") or "Not stated", muted=not rec.get("people"))
        conf = rec.get("confidence")
        fact(c[3], "Confidence", (f"{conf:.0%} (calibrated model)" if "ML" in method else f"{conf:.0%} (AI estimate)")
             if isinstance(conf, (int, float)) else "Not measured",
             muted=not isinstance(conf, (int, float)))
        st.markdown(f"**{SEVERITY_GUIDE.get(sev, '')}.** Why: {rec.get('severity_reason', 'not given')}.")
        st.markdown(f"**Summary:** {rec.get('summary', '')}")
        if rec.get("is_emergency") and not rec.get("location"):
            st.warning("No known landmark in this call. Call the reporter back to get the location; "
                       "it will not be dispatched or drawn on the map until then.")
        for note in rec.get("model_notes", []):
            st.caption(f"🤖 {note[0].upper() + note[1:]}.")
        if rec.get("needs_review") and rec.get("is_emergency"):
            st.warning(f"🧑‍💼 Please check this call: "
                       + (f"the rules say {rec['incident_type']} but the model says {rec['model_type']}." if not rec.get("model_agrees")
                          else f"the model is only {rec['confidence']:.0%} sure."))
        if method == "Rule-based fallback":
            st.caption("Rule-based triage: add a Groq API key in the sidebar for AI reading of messy or unclear calls.")
        if rec.get("Recording"):
            st.markdown("🎧 **Caller’s recording**")
            st.audio(rec["Recording"], format="audio/wav")
        with st.expander("Raw data (for developers)"):
            st.json({k: v for k, v in rec.items() if k not in ("Call text", "Recording")})
            if "model_type" in rec:
                st.caption("Model details: see MODEL_CARD.md for how it was trained and tested.")

# ----------------------- Learned triage model (exported by notebook Part 8) -----------------------
MODEL_FILE = Path(__file__).parent / "triage_model.json"


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified: float) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    for part in ("type_model", "severity_model"):                      # pre-compute arrays once
        mm = m[part]
        mm["_W"], mm["_b"] = [list(r) for r in mm["coef"]], list(mm["intercept"])
    return m


def _ngram_counts(text: str, vocab: dict, lo: int, hi: int, stats: dict | None = None) -> Counter:
    """Exactly scikit-learn's char_wb analyzer (lowercase, words padded with spaces)."""
    counts, total = Counter(), 0
    for w in re.sub(r"\s\s+", " ", text.lower()).split():
        w = f" {w} "
        for n in range(lo, hi + 1):
            off, grams = 0, [w[0:n]]
            while off + n < len(w):
                off += 1
                grams.append(w[off:off + n])
            total += len(grams)
            for g in grams:
                j = vocab.get(g)
                if j is not None:
                    counts[j] += 1
            if off == 0:
                break
    if stats is not None:
        stats["familiarity"] = sum(counts.values()) / total if total else 0.0
    return counts


def model_proba(part: dict, text: str, stats: dict | None = None) -> dict[str, float]:
    """Sublinear TF-IDF -> L2 norm -> logistic regression -> temperature-scaled softmax."""
    counts = _ngram_counts(text, part["vocab"], *part["ngram_range"], stats=stats)
    x = {j: (1 + math.log(c)) * part["idf"][j] for j, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in x.values())) or 1.0
    z = [(sum(row[j] * v for j, v in x.items()) / norm + b) / part["temperature"] for row, b in zip(part["_W"], part["_b"])]
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    tot = sum(e)
    return {c: v / tot for c, v in zip(part["classes"], e)}


def apply_model(triage: dict, call_text: str, model: dict) -> dict:
    """Safety hybrid from notebook Part 8: rules keep the type; the model rescues calls the rules could not type,
    can raise severity (never lower it), gives a calibrated confidence, and flags disagreement for a human."""
    info: dict = {}
    tp = model_proba(model["type_model"], call_text, stats=info)
    # Calibration only holds for wording like the training data. For unfamiliar wording, shrink confidence
    # towards "no idea" in proportion to how unfamiliar it is, and never claim more than 99%.
    fam = model.get("familiarity", {})
    ref = fam.get("p05", 0.0) or 0.0
    trust = 1.0 if not ref else min(1.0, info.get("familiarity", 0.0) / ref)
    k = len(tp)
    tp = {c: min(0.99, trust * p + (1 - trust) / k) for c, p in tp.items()}
    app_p: dict[str, float] = {}
    for cls, p in tp.items():
        app_p[model["app_type_map"][cls]] = app_p.get(model["app_type_map"][cls], 0.0) + p
    m_type = max(app_p, key=app_p.get)
    thr = model.get("review_threshold", 0.7)
    out = dict(triage)
    notes = []
    confident = app_p[m_type] >= thr
    if out.get("is_emergency") and m_type != "Noise / prank" and m_type != out["incident_type"] and confident and (
            model.get("type_leader") == "model" or not out.get("type_from_keywords")):
        # the leader (chosen on development data in Part 8) decides; the model never dismisses a call on its own
        notes.append(f"type from the model; the rules said {out['incident_type']}"
                     + ("" if out.get("type_from_keywords") else " only as a default"))
        out["incident_type"] = m_type
    if out.get("is_emergency"):
        sp = model_proba(model["severity_model"], call_text)
        m_sev = int(max(sp, key=sp.get))
        if m_sev > int(out["severity"]):
            notes.append(f"model raised severity from {out['severity']} to {m_sev}")
            out["severity_reason"] = f"{out.get('severity_reason', '')}; the model rates calls like this at {m_sev} ({sp[str(m_sev)]:.0%} sure)".lstrip("; ")
            out["severity"] = m_sev
    final = out["incident_type"] if out.get("is_emergency") else "Noise / prank"
    out["confidence"] = round(app_p.get(final, 0.0), 3)
    out["model_type"] = m_type
    out["model_agrees"] = (m_type == final)
    out["familiarity"] = round(info.get("familiarity", 0.0), 3)
    out["needs_review"] = (not out["model_agrees"]) or out["confidence"] < thr
    if trust < 1.0:
        notes.append(f"unfamiliar wording ({out['familiarity']:.0%} of its patterns seen in training), so confidence was lowered")
    out["model_notes"] = notes
    out["triage_method"] = "Rules + ML model"
    return out


def run_triage(call_text: str) -> tuple[dict, list[tuple[str, str]]]:
    """Groq AI if switched on and configured, otherwise rules (+ the learned model). Returns (triage, messages)."""
    msgs, triage = [], None
    if use_groq and groq_key:
        try:
            triage = groq_triage(call_text, groq_key, groq_model)
            triage["triage_method"] = "Groq AI"
        except Exception as exc:
            msgs.append(("warning", f"Groq triage failed; using rule-based fallback. Details: {exc}"))
    elif use_groq and not groq_key:
        msgs.append(("info", "Groq AI is switched on but no API key is set, so keyword rules were used."))
    if triage is None:
        triage = rule_based_triage(call_text)
        triage["triage_method"] = "Rule-based fallback"
        if use_model and triage_model is not None:
            triage = apply_model(triage, call_text, triage_model)
    return triage, msgs


# ----------------------- Emergency Help (native citizen page) -----------------------
HELPLINES = [("112", "All emergencies"), ("108", "Ambulance"), ("101", "Fire"), ("100", "Police"),
             ("1091", "Women helpline"), ("1098", "Child helpline")]
HELP_TYPES = {   # button -> (dispatch type, words added to the report so triage understands it)
    "🔥 Fire": ("Fire", "fire"),
    "🚑 Medical": ("Medical", "medical emergency"),
    "🚗 Road accident": ("Medical", "road accident, people injured"),
    "🏚️ Building collapse": ("Rescue", "building collapse"),
    "🌊 Flood": ("Rescue", "flood water rising"),
    "💨 Gas leak": ("Fire", "gas leak, strong gas smell"),
}
DANGER_SIGNS = {"Someone is trapped": 5, "Someone is not breathing or unconscious": 5,
                "Heavy bleeding": 4, "Fire or danger is spreading": 4, "Someone is injured": 3}
WHILE_YOU_WAIT = {
    "Fire": ["Get everyone out and stay out. Do not go back inside for belongings.",
             "If there is smoke, stay low, close doors behind you, and do not use lifts.",
             "If gas is leaking: no flames or switches; open doors and windows only if it is safe, then leave."],
    "Medical": ["Stay with the person and keep them still, warm and calm. Do not give food or drink.",
                "If they are not breathing normally and you are trained, start CPR; otherwise follow the 112 operator's instructions.",
                "Press firmly on heavy bleeding with a clean cloth and keep pressing.",
                "After a road accident, switch on hazard lights and do not move injured people unless they are in danger."],
    "Rescue": ["Move away from damaged buildings, walls and electric wires. Do not go inside.",
               "If someone is trapped, keep talking to them and tap on pipes or walls so rescuers can find them.",
               "In a flood, move to higher ground and never walk or drive through moving water."],
}
HELP_STAGES = ["Report received", "Help assigned", "On the way", "Arrived", "Completed"]
SCENE_MINUTES = 10


def dispatch_help(rep_: dict) -> None:
    """Assign the fastest free vehicle of the right type on today's roads (respects road closures and other jobs)."""
    if not rep_.get("location") or rep_.get("vehicle"):
        return
    graph = add_closures(make_city_graph(), closure_rows, closure_cols)
    available = [v for v in VEHICLES if v["Vehicle ID"] not in st.session_state.used_vehicle_ids]
    vehicle, minutes = choose_vehicle(rep_["type"], rep_["location"], graph, available)
    if vehicle is None:
        rep_["status"] = "queued"
        return
    st.session_state.used_vehicle_ids.add(vehicle["Vehicle ID"])
    rep_.update(vehicle=vehicle["Vehicle ID"], vehicle_type=vehicle["Type"], eta=max(1.0, minutes), assigned_at=time.time(),
                route=route_nodes(graph, vehicle["Station"], LANDMARKS[rep_["location"]]), status="assigned")
    for c in st.session_state.calls:
        if c.get("Call ID") == rep_["call_id"]:
            c["Help vehicle"] = vehicle["Vehicle ID"]


def help_progress(rep_: dict) -> dict:
    """Where the help is now, from the simulated clock (one simulated minute = `sim_seconds` real seconds)."""
    if not rep_.get("assigned_at"):
        return {"stage": 0, "remaining": None, "frac": 0.0}
    elapsed = (time.time() - rep_["assigned_at"]) / sim_seconds
    if elapsed < rep_["eta"]:
        return {"stage": 2 if elapsed > 0.2 else 1, "remaining": rep_["eta"] - elapsed, "frac": elapsed / rep_["eta"]}
    if elapsed < rep_["eta"] + SCENE_MINUTES:
        return {"stage": 3, "remaining": 0.0, "frac": 1.0}
    if rep_["vehicle"] in st.session_state.used_vehicle_ids and not rep_.get("released"):
        st.session_state.used_vehicle_ids.discard(rep_["vehicle"])          # vehicle is free for new jobs
        rep_["released"] = True
    return {"stage": 4, "remaining": 0.0, "frac": 1.0}


def draw_help_map(rep_: dict, frac: float):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_xlim(-0.5, GRID_SIZE - 0.5); ax.set_ylim(-0.5, GRID_SIZE - 0.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for k in range(GRID_SIZE):
        ax.axhline(k, color="#cbd4dc", lw=0.6, zorder=0); ax.axvline(k, color="#cbd4dc", lw=0.6, zorder=0)
    ax.axvspan(9.15, 9.85, color="#c3dcec", zorder=0.5)                   # river between x=9 and x=10
    for y in (3, 10, 16):
        ax.plot([9, 10], [y, y], color="#5b6b7a", lw=3, zorder=1)          # bridges
    for y in closure_rows:
        ax.axhline(y, color="#d62839", lw=2, ls="--", alpha=.6)
    for x in closure_cols:
        ax.axvline(x, color="#d62839", lw=2, ls="--", alpha=.6)
    route = rep_.get("route") or []
    if route:
        xs, ys = zip(*route)
        ax.plot(xs, ys, color="#1c5d99", lw=3, alpha=.85, zorder=2)
        pos = route[min(len(route) - 1, int(round(frac * (len(route) - 1))))]
        ax.scatter(*pos, s=180, marker="s", color="white", edgecolors="#d62839", linewidths=2.5, zorder=4)
        ax.annotate(rep_["vehicle"], pos, xytext=(6, 6), textcoords="offset points", fontsize=8, fontweight="bold")
    if rep_.get("location"):
        tx, ty = LANDMARKS[rep_["location"]]
        ax.scatter(tx, ty, s=260, color=SEV_COLOR.get(rep_["severity"], "#d62839"), edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(rep_["location"], (tx, ty), xytext=(6, -12), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------- Speech to text (works even when browser voice typing does not) -----------------------
VOICE_LANGS = {"English (India)": "en-IN", "हिन्दी (Hindi)": "hi-IN"}
WHISPER_HINT = ("Emergency call in an Indian city. Words that may appear: fire, smoke, accident, injured, trapped, "
                "building collapse, flood, gas leak, ambulance, hospital, police station, bus stand, railway station, "
                "market, mandir, masjid, gurudwara, chowk, thana.")


def transcribe_audio(wav: bytes, lang: str, api_key: str, whisper_model: str) -> tuple[str, str]:
    """Turn a WAV recording into text. Groq Whisper first (if a key is set), then free Google speech.

    Returns (text, engine name). Raises RuntimeError with a readable reason when nothing worked.
    """
    problems = []
    if api_key and Groq is not None:
        try:
            result = Groq(api_key=api_key).audio.transcriptions.create(
                file=("report.wav", wav), model=whisper_model, language=lang[:2], prompt=WHISPER_HINT, temperature=0.0)
            text = (getattr(result, "text", "") or "").strip()
            if text:
                return text, "Groq Whisper"
            problems.append("Groq Whisper heard no words")
        except Exception as exc:
            problems.append(f"Groq Whisper failed ({str(exc)[:120]})")
    if sr is not None:
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(wav)) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_google(audio, language=lang).strip(), "Google speech"
        except sr.UnknownValueError:
            problems.append("no words could be recognised: speak closer to the microphone and a little slower")
        except Exception as exc:
            problems.append(f"the free speech service failed ({str(exc)[:120]})")
    else:
        problems.append("the SpeechRecognition package is not installed")
    raise RuntimeError("; ".join(problems))


def voice_recorder(key: str, label: str, api_key: str, whisper_model: str) -> dict | None:
    """Record with Streamlit's own microphone widget and transcribe each new recording once."""
    lang_name = st.radio("Language you will speak", list(VOICE_LANGS), horizontal=True, key=f"{key}_lang")
    audio = st.audio_input(label, key=f"{key}_audio")
    if audio is None:
        return st.session_state.get(f"{key}_result")
    data = audio.getvalue()
    digest = hashlib.sha1(data).hexdigest()
    if st.session_state.get(f"{key}_hash") != digest:          # a new recording: transcribe it once
        st.session_state[f"{key}_hash"] = digest
        with st.spinner("Turning your voice into text…"):
            try:
                text, engine = transcribe_audio(data, VOICE_LANGS[lang_name], api_key, whisper_model)
                st.session_state[f"{key}_result"] = {"text": text, "lang": VOICE_LANGS[lang_name], "engine": engine, "new": True,
                                                     "audio": data}
            except RuntimeError as exc:
                st.session_state[f"{key}_result"] = None
                st.error(f"Could not turn the recording into text: {exc}. You can record again or type instead.")
                return None
    return st.session_state.get(f"{key}_result")


# ----------------------- Sidebar -----------------------
st.sidebar.header("Configuration")
use_groq = st.sidebar.toggle("Use Groq AI for triage", value=True)
groq_key = st.sidebar.text_input(
    "Groq API key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password",
    help="Or add GROQ_API_KEY to Streamlit Secrets. Do not commit your key to GitHub.",
)
groq_model = st.sidebar.text_input("Groq model", value="llama-3.3-70b-versatile")
st.sidebar.caption("If Groq is disabled or unavailable, rule-based triage is used.")
triage_model = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime if MODEL_FILE.exists() else 0.0)
use_model = st.sidebar.toggle("Use the learned triage model", value=triage_model is not None, disabled=triage_model is None,
                              help="Trained and checked in notebook Part 8 (see MODEL_CARD.md). Rules keep the type; the model "
                                   "adds a calibrated confidence, can raise severity, and flags calls for review.")
if triage_model is None:
    st.sidebar.caption("No triage_model.json next to app.py: rules only. Run notebook Part 8 to create it.")
else:
    _m = triage_model.get("metrics_hidden_test", {})
    st.sidebar.caption(f"Model {triage_model.get('version', '')}: on the hidden test, type macro-F1 "
                       f"{_m.get('type macro-F1', float('nan')):.2f}, severe under-triage {_m.get('severe under-triaged', float('nan')):.0%}.")
whisper_model = st.sidebar.selectbox("Speech-to-text model (Groq)", ["whisper-large-v3-turbo", "whisper-large-v3"],
                                     help="Used for recordings when a Groq key is set. Without a key, free Google speech recognition is used.")
st.sidebar.divider()
st.sidebar.subheader("Simulated road closures")
closure_rows = st.sidebar.multiselect("Close horizontal street row(s)", options=list(range(GRID_SIZE)), default=[])
closure_cols = st.sidebar.multiselect("Close vertical street column(s)", options=list(range(GRID_SIZE)), default=[])
st.sidebar.divider()
st.sidebar.subheader("Emergency Help")
sim_seconds = st.sidebar.slider("Demo speed: real seconds per simulated minute", 0.5, 10.0, 2.0, step=0.5,
                                help="Lower = help arrives faster in the demo. Travel times come from the road map.")
st.sidebar.subheader("Sahayak 112 app")
sahayak_height = st.sidebar.slider("App height (pixels)", 700, 2400, 1500, step=100,
                                   help="Make it taller to see the response tracking and feedback panels without scrolling inside the app.")

if "calls" not in st.session_state:
    st.session_state.calls = []
if "dispatches" not in st.session_state:
    st.session_state.dispatches = []
if "used_vehicle_ids" not in st.session_state:
    st.session_state.used_vehicle_ids = set()
if "help_reports" not in st.session_state:
    st.session_state.help_reports = []

tab_help, tab0, tab1, tab2, tab3 = st.tabs(["🆘 Emergency Help", "🧭 Sahayak 112 (full simulation)", "📞 Triage a call",
                                             "🚑 Dispatch board", "🗺️ City map"])

with tab_help:
    st.markdown(
        "<div style='background:linear-gradient(135deg,#D62839,#A51D2D);color:#fff;border-radius:18px;padding:18px 22px;margin-bottom:6px'>"
        "<div style='font-size:1.6rem;font-weight:800'>🆘 Need help now?</div>"
        "<div style='opacity:.92'>Tell us what happened and where. We find the fastest vehicle on today’s roads and show you when it will arrive.</div>"
        "</div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 4px'>" + "".join(
        f"<a href='tel:{n}' style='text-decoration:none;border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:6px 12px;"
        f"color:inherit'><b style='color:#D62839'>📞 {n}</b> <span style='opacity:.8'>{label}</span></a>" for n, label in HELPLINES)
        + "</div>", unsafe_allow_html=True)
    st.caption("This page is a training simulation: no real vehicle is sent. In a real emergency in India, call 112.")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### 1. What happened?")
            h_kind = st.pills("Type of emergency", list(HELP_TYPES), key="help_kind", label_visibility="collapsed")
            st.markdown("#### 2. Where are you?")
            h_place = st.selectbox("Nearest landmark", ["Detect from my words"] + sorted(LANDMARKS), key="help_place")
            st.markdown("#### 3. Tell us more")
            with st.expander("🎙️ Speak instead of typing", expanded=False):
                h_voice = voice_recorder("help", "Record what is happening", groq_key, whisper_model)
                if h_voice and h_voice.get("new"):
                    st.session_state.help_text = h_voice["text"]
                    h_voice["new"] = False
                if h_voice:
                    st.success(f"Heard ({h_voice['engine']}): “{h_voice['text']}”")
                    st.audio(h_voice["audio"], format="audio/wav")
            h_text = st.text_area("What is happening? (optional)", key="help_text", height=80,
                                  placeholder="For example: smoke from the second floor, my neighbour is still inside")
            h_signs = st.pills("Danger signs", list(DANGER_SIGNS), selection_mode="multi", key="help_signs")
            h_people = st.number_input("How many people need help?", min_value=0, max_value=200, value=0, step=1, key="help_people",
                                       help="Leave at 0 if you don't know.")
            send = st.button("🚨 Send for help", type="primary")
        if send:
            words = HELP_TYPES[h_kind][1] if h_kind else ""
            place = None if h_place == "Detect from my words" else h_place
            report_text = ". ".join(x for x in [words, (h_text or "").strip(), f"near {place}" if place else "",
                                                ", ".join(h_signs or [])] if x)
            if not report_text:
                st.error("Choose what happened, or say or type a few words, then send.")
            else:
                triage, msgs = run_triage(report_text)
                for kind_, m_ in msgs:
                    getattr(st, kind_)(m_)
                triage["is_emergency"] = True                       # someone pressed "Send for help": never dismiss it
                if h_kind:
                    triage["incident_type"] = HELP_TYPES[h_kind][0]
                elif triage.get("incident_type") not in ("Fire", "Rescue", "Medical"):
                    triage["incident_type"] = "Medical"
                if h_signs:
                    top = max(DANGER_SIGNS[sg] for sg in h_signs)
                    if top > int(triage.get("severity", 1)):
                        triage["severity"], triage["severity_reason"] = top, f"you said: {', '.join(h_signs).lower()}"
                if place:
                    triage["location"] = place
                if h_people:
                    triage["people"] = int(h_people)
                call_id = f"CALL-{len(st.session_state.calls) + 1:03d}"
                ref = f"HELP-{len(st.session_state.help_reports) + 1:03d}"
                voiced = bool(h_voice) and (h_text or "").strip() == h_voice.get("text", "")
                st.session_state.calls.append({"Call ID": call_id, "Call text": report_text, "Source": "Emergency Help",
                                               "Voice engine": h_voice["engine"] if voiced else None,
                                               "Recording": h_voice["audio"] if voiced else None, "help_ref": ref, **triage})
                rep_ = {"ref": ref, "call_id": call_id, "created": time.time(), "type": triage["incident_type"],
                        "severity": int(triage["severity"]), "location": triage.get("location"), "people": triage.get("people"),
                        "reason": triage.get("severity_reason", ""), "method": triage.get("triage_method", ""),
                        "text": report_text, "status": "received" if triage.get("location") else "needs_location",
                        "vehicle": None, "feedback": None}
                dispatch_help(rep_)
                st.session_state.help_reports.append(rep_)
                st.session_state.help_track = ref
                st.toast(f"Report {ref} received. Help is being arranged.", icon="🚨")

    with right:
        reports = st.session_state.help_reports
        if not reports:
            with st.container(border=True):
                st.markdown("#### Your help request")
                st.info("After you send a report, its reference number, the vehicle on its way and the arrival time appear here.")
        else:
            refs = [r_["ref"] for r_ in reports][::-1]                 # newest first
            track_ref = st.selectbox("Track a report", refs, key="help_track")
            rep_ = next(r_ for r_ in reports if r_["ref"] == track_ref)

            if rep_["status"] == "needs_location":
                with st.container(border=True):
                    st.warning(f"**{rep_['ref']}**: we could not find your location. Which landmark is closest?")
                    fix = st.selectbox("Nearest landmark", sorted(LANDMARKS), key=f"fix_{rep_['ref']}")
                    if st.button("Set location and send help", key=f"fixbtn_{rep_['ref']}"):
                        rep_["location"], rep_["status"] = fix, "received"
                        for c in st.session_state.calls:
                            if c.get("Call ID") == rep_["call_id"]:
                                c["location"] = fix
                        dispatch_help(rep_)
                        st.rerun()

            @st.fragment(run_every=1.0)
            def live_tracking(ref_: str) -> None:
                r_ = next(x for x in st.session_state.help_reports if x["ref"] == ref_)
                if r_["status"] == "queued":
                    dispatch_help(r_)                               # retry: a vehicle may have become free
                pr = help_progress(r_)
                with st.container(border=True):
                    head = st.columns([3, 1])
                    title = ["Help is being arranged", "Help assigned", "Help is on the way", "Help has arrived", "Completed"][pr["stage"]]
                    if r_["status"] == "queued":
                        title = "Waiting for a free vehicle"
                    head[0].markdown(f"### {title}")
                    head[1].markdown(f"<div style='text-align:right;margin-top:12px'><code>{r_['ref']}</code></div>", unsafe_allow_html=True)
                    if pr["stage"] in (1, 2):
                        st.markdown(f"<div style='font-size:3rem;font-weight:800;line-height:1'>{max(1, math.ceil(pr['remaining']))}"
                                    f"<span style='font-size:1rem;font-weight:500;opacity:.75'> min until {r_['vehicle']} arrives</span></div>",
                                    unsafe_allow_html=True)
                        st.progress(min(1.0, pr["frac"]))
                    elif pr["stage"] >= 3:
                        st.success(f"{r_['vehicle']} reached {r_['location']} after {r_['eta']:.0f} simulated minutes.")
                    elif r_["status"] == "queued":
                        st.warning("All suitable vehicles are busy. You are in the queue; the first free one will be sent.")
                    steps = st.columns(len(HELP_STAGES))
                    for k_, (col, name) in enumerate(zip(steps, HELP_STAGES)):
                        done = k_ <= pr["stage"] and (k_ == 0 or r_.get("vehicle"))
                        col.markdown(f"<div style='text-align:center;font-size:.8rem'><div style='font-size:1.2rem'>{'🟢' if done else '⚪'}</div>{name}</div>",
                                     unsafe_allow_html=True)
                    f1, f2, f3 = st.columns(3)
                    f1.markdown(f"**Emergency**  \n{r_['type']}")
                    f2.markdown(f"**Severity**  \n<span style='color:{SEV_COLOR.get(r_['severity'])};font-weight:800'>{r_['severity']} of 5</span>",
                                unsafe_allow_html=True)
                    f3.markdown(f"**Location**  \n{r_['location'] or 'not found yet'}")
                    if r_.get("route"):
                        fig_ = draw_help_map(r_, pr["frac"])
                        st.pyplot(fig_, width="stretch")
                        plt.close(fig_)
                if pr["stage"] >= 3 and not r_.get("_shown_done"):
                    r_["_shown_done"] = True
                    st.rerun(scope="app")                           # show the feedback form below

            live_tracking(track_ref)

            with st.container(border=True):
                st.markdown("#### While you wait")
                for tip in WHILE_YOU_WAIT.get(rep_["type"], WHILE_YOU_WAIT["Medical"]):
                    st.markdown(f"- {tip}")
                st.caption("General safety advice only. Always follow the instructions of the emergency operator.")

            if rep_.get("_shown_done") or help_progress(rep_)["stage"] >= 3:
                with st.container(border=True):
                    if rep_.get("feedback"):
                        fb = rep_["feedback"]
                        st.success(f"Thank you. You rated this response {'★' * fb['rating']} ({fb['rating']} of 5).")
                    else:
                        st.markdown("#### How was the response?")
                        stars = st.feedback("stars", key=f"stars_{rep_['ref']}")
                        tags = st.pills("What stood out?", ["Fast response", "Crew was helpful", "Kept me informed",
                                                           "Took too long", "Hard to explain location", "App was hard to use"],
                                        selection_mode="multi", key=f"tags_{rep_['ref']}")
                        comment = st.text_input("Anything else? (optional)", key=f"cmt_{rep_['ref']}")
                        if st.button("Send feedback", key=f"fbbtn_{rep_['ref']}"):
                            if stars is None:
                                st.error("Choose 1 to 5 stars first.")
                            else:
                                rep_["feedback"] = {"rating": stars + 1, "tags": tags or [], "comment": comment}
                                st.rerun()

        if reports:
            with st.expander(f"📊 Help requests this session ({len(reports)})"):
                rated = [r_ for r_ in reports if r_.get("feedback")]
                m1, m2, m3 = st.columns(3)
                m1.metric("Requests", len(reports))
                m2.metric("Average travel time", f"{sum(r_['eta'] for r_ in reports if r_.get('eta')) / max(1, sum(1 for r_ in reports if r_.get('eta'))):.1f} min")
                m3.metric("Average rating", f"{sum(r_['feedback']['rating'] for r_ in rated) / len(rated):.1f} ★" if rated else "–")
                table = pd.DataFrame([{"Reference": r_["ref"], "Emergency": r_["type"], "Severity": r_["severity"],
                                       "Location": r_["location"] or "Unknown", "Vehicle": r_["vehicle"] or "waiting",
                                       "Travel (sim min)": round(r_["eta"], 1) if r_.get("eta") else None,
                                       "Rating": r_["feedback"]["rating"] if r_.get("feedback") else None,
                                       "Tags": "; ".join(r_["feedback"]["tags"]) if r_.get("feedback") else "",
                                       "Comment": r_["feedback"]["comment"] if r_.get("feedback") else ""} for r_ in reports])
                st.dataframe(table, width="stretch", hide_index=True)
                st.download_button("Download help requests and feedback (CSV)", table.to_csv(index=False).encode("utf-8"),
                                   file_name="help_requests.csv", mime="text/csv")

with tab0:
    page = load_sahayak(str(SAHAYAK_FILE), SAHAYAK_FILE.stat().st_mtime if SAHAYAK_FILE.exists() else 0.0)
    if page is None:
        st.error("sahayak112.html was not found. Put it in the same folder as app.py "
                 "(on Streamlit Cloud: in the same GitHub repository), then reload this page.")
    else:
        st.caption("Report an emergency by voice or text, track help live, then rate the response. "
                   "This app runs its own simulation, separate from the other tabs.")
        with st.expander("🎙️ “Tap to speak” not working? Record your report here instead", expanded=False):
            st.caption("Some browsers and computers block voice typing inside an embedded page. This recorder uses "
                       "Streamlit’s own microphone and turns your voice into text on the server, "
                       + ("with Groq Whisper." if groq_key else "with free Google speech recognition (add a Groq key for Whisper, which handles Hindi and noise better).")
                       + " Your words are then placed in the report box of the app below; check them and press Send report.")
            heard = voice_recorder("sah", "Record your emergency report", groq_key, whisper_model)
            if heard:
                st.success(f"Heard ({heard['engine']}): “{heard['text']}”")
                st.markdown("🎧 **Listen to what you recorded**")
                st.audio(heard["audio"], format="audio/wav")
                st.caption("The app below has reloaded with these words in the report box. A new recording replaces them.")
        prefill = st.session_state.get("sah_result")
        html = page
        if prefill:
            payload = json.dumps({k: prefill[k] for k in ("text", "lang", "engine")}, ensure_ascii=False).replace("</", "<\\/")
            html = page.replace("/*SAHAYAK_PREFILL*/null/*END_SAHAYAK_PREFILL*/", f"/*SAHAYAK_PREFILL*/{payload}/*END_SAHAYAK_PREFILL*/")
        components.html(html, height=sahayak_height, scrolling=True)
        st.download_button("Download Sahayak 112 (open it in any browser, no Python needed)",
                           page.encode("utf-8"), file_name="sahayak112.html", mime="text/html")

with tab1:
    st.subheader("Enter an incoming call")
    sample = st.selectbox("Try a sample call", [
        "Caller reports a serious road accident with two injured people near Central Hospital.",
        "Smoke and flames are coming from a building near the Shopping Mall.",
        "A person is unconscious and not breathing near Railway Station.",
        "Several people are trapped after a building collapse in the Industrial Area.",
        "This is a prank call; there is no emergency.",
        "A bus crashed near River Bridge and people need help.",
    ])
    if st.session_state.get("last_sample") != sample:          # picking a sample replaces the transcript
        st.session_state.last_sample = sample
        st.session_state.call_text = sample
        st.session_state.call_source = "Sample"
        st.session_state.call_audio = None
    with st.expander("🎙️ Speak your complaint instead of typing", expanded=True):
        auto = st.checkbox("Triage automatically as soon as my words are written", value=True, key="tri_auto")
        spoken = voice_recorder("tri", "Record your complaint", groq_key, whisper_model)
        if spoken and spoken.get("new"):
            st.session_state.call_text = spoken["text"]
            st.session_state.call_source = "Voice"
            st.session_state.call_engine = spoken["engine"]
            st.session_state.call_audio = spoken["audio"]
            st.session_state.auto_triage = auto
            spoken["new"] = False
        if spoken:
            st.success(f"Heard ({spoken['engine']}): “{spoken['text']}”. It is now in the transcript box.")
            st.markdown("🎧 **Listen to what you recorded**")
            st.audio(spoken["audio"], format="audio/wav")
            if spoken["lang"] == "hi-IN" and not groq_key:
                st.caption("Keyword rules understand English best. Add a Groq key for AI triage of Hindi calls.")
    call_text = st.text_area("Call transcript", key="call_text", height=110,
                             on_change=lambda: st.session_state.update(
                                 call_source="Voice, edited" if str(st.session_state.get("call_source", "")).startswith("Voice") else "Typed"))
    clicked = st.button("Triage call", type="primary")
    if clicked or st.session_state.pop("auto_triage", False):
        if not call_text.strip():
            st.error("Enter a call transcript first.")
        else:
            triage, msgs = run_triage(call_text)
            for kind_, m_ in msgs:
                getattr(st, kind_)(m_)
            record = {
                "Call ID": f"CALL-{len(st.session_state.calls) + 1:03d}",
                "Call text": call_text.strip(),
                "Source": st.session_state.get("call_source", "Typed"),
                "Voice engine": st.session_state.get("call_engine") if str(st.session_state.get("call_source", "")).startswith("Voice") else None,
                "Recording": st.session_state.get("call_audio") if str(st.session_state.get("call_source", "")).startswith("Voice") else None,
                **triage,
            }
            st.session_state.calls.append(record)
            st.success("Call triaged.")
    if st.session_state.calls:
        show_triage_card(st.session_state.calls[-1])          # stays visible after other clicks
    if st.session_state.calls:
        st.markdown("#### Triage history")
        hist = pd.DataFrame(st.session_state.calls)
        show_cols = ["Call ID", "incident_type", "severity", "location", "is_emergency", "triage_method", "confidence"]
        hist["confidence"] = hist["confidence"].map(lambda v: f"{v:.0%}" if isinstance(v, (int, float)) and v == v else "Not measured")
        hist["location"] = hist["location"].fillna("Unknown")
        hist = hist.rename(columns={
            "incident_type": "Incident type", "severity": "Severity",
            "location": "Location", "is_emergency": "Emergency?",
            "triage_method": "Triage method", "confidence": "Confidence"
        })
        st.dataframe(hist[[c for c in ["Call ID", "Source", "Incident type", "Severity", "Location", "Emergency?", "Triage method", "Confidence"] if c in hist.columns]], width="stretch")
        voiced = [c for c in st.session_state.calls if c.get("Recording")]
        if voiced:
            with st.expander(f"🎧 Listen to voice complaints ({len(voiced)})"):
                for c in reversed(voiced):
                    st.markdown(f"**{c['Call ID']}**: {c['incident_type']}, severity {c['severity']}, "
                                f"{c.get('location') or 'location unknown'}. “{c['Call text']}”")
                    st.audio(c["Recording"], format="audio/wav")

with tab2:
    st.subheader("Prioritized incident and dispatch board")
    emergency_calls = [c for c in st.session_state.calls if c.get("is_emergency")]
    # A simple duplicate grouping by exact normalized location + incident type.
    grouped = {}
    for call in emergency_calls:
        key = (call.get("location"), call.get("incident_type"))
        if key not in grouped:
            grouped[key] = {
                "Incident ID": f"INC-{len(grouped) + 1:03d}",
                "Location": call.get("location"),
                "Incident type": call.get("incident_type"),
                "Severity": int(call.get("severity", 3)),
                "Calls": 1,
                "Summary": call.get("summary", ""),
                "Call IDs": [call.get("Call ID")],
                "Help vehicle": call.get("Help vehicle"),
            }
        else:
            grouped[key]["Calls"] += 1
            grouped[key]["Severity"] = max(grouped[key]["Severity"], int(call.get("severity", 3)))
            grouped[key]["Call IDs"].append(call.get("Call ID"))
            grouped[key]["Help vehicle"] = grouped[key].get("Help vehicle") or call.get("Help vehicle")
    incidents = sorted(grouped.values(), key=lambda i: (-i["Severity"], i["Incident ID"]))
    if not incidents:
        st.info("Triage an emergency call first to populate the dispatch board.")
    else:
        board = pd.DataFrame(incidents)
        board["Location"] = board["Location"].fillna("Unknown: call back")
        st.dataframe(board[["Incident ID", "Location", "Incident type", "Severity", "Calls", "Summary"]], width="stretch")
        if st.button("Dispatch available vehicles", type="primary"):
            graph = add_closures(make_city_graph(), closure_rows, closure_cols)
            available = [v.copy() for v in VEHICLES if v["Vehicle ID"] not in st.session_state.used_vehicle_ids]
            results = []
            for inc in incidents:
                if inc.get("Help vehicle"):
                    results.append({"Incident ID": inc["Incident ID"], "Location": inc["Location"], "Severity": inc["Severity"],
                                    "Vehicle": inc["Help vehicle"], "Estimated travel (simulated min)": None,
                                    "Status": "Handled from Emergency Help (vehicle already sent)"})
                    continue
                assigned, minutes = choose_vehicle(inc["Incident type"], inc["Location"], graph, available)
                if assigned is None:
                    results.append({
                        "Incident ID": inc["Incident ID"], "Location": inc["Location"] or "Unknown",
                        "Severity": inc["Severity"], "Vehicle": "UNASSIGNED",
                        "Estimated travel (simulated min)": None,
                        "Status": "Needs location: call the reporter back" if not inc["Location"] else "No available reachable vehicle",
                    })
                    continue
                # Claim the vehicle exactly once for this dispatch batch.
                st.session_state.used_vehicle_ids.add(assigned["Vehicle ID"])
                available = [v for v in available if v["Vehicle ID"] != assigned["Vehicle ID"]]
                results.append({
                    "Incident ID": inc["Incident ID"], "Location": inc["Location"],
                    "Severity": inc["Severity"], "Vehicle": assigned["Vehicle ID"],
                    "Estimated travel (simulated min)": round(minutes, 1), "Status": "Assigned"
                })
            st.session_state.dispatches = results
            st.success("Dispatch batch completed. Assigned vehicles cannot be reused in this session.")
        if st.session_state.dispatches:
            st.markdown("#### Latest dispatch result")
            st.dataframe(pd.DataFrame(st.session_state.dispatches), width="stretch")
            st.download_button(
                "Download dispatch results (CSV)",
                pd.DataFrame(st.session_state.dispatches).to_csv(index=False).encode("utf-8"),
                file_name="dispatch_results.csv",
                mime="text/csv",
            )
    if st.button("Reset demo state"):
        st.session_state.calls = []
        st.session_state.dispatches = []
        st.session_state.used_vehicle_ids = set()
        st.rerun()

with tab3:
    st.subheader("Simulated city map")
    st.caption("A grid simulation only; it does not represent real roads, GPS, or live traffic.")
    graph = add_closures(make_city_graph(), closure_rows, closure_cols)
    fig = draw_map(st.session_state.calls, st.session_state.dispatches, closure_rows, closure_cols)
    st.pyplot(fig, width="stretch")
    plt.close(fig)                      # free memory: Streamlit reruns this script on every click
    st.markdown("#### Landmark directory")
    st.dataframe(
        pd.DataFrame([{"Landmark": name, "Grid X": xy[0], "Grid Y": xy[1]} for name, xy in LANDMARKS.items()]),
        width="stretch",
    )

st.divider()
st.caption("Audit note: this demo displays triage method, confidence, call history, incident grouping, and simulated dispatch estimates. Verify all logic and safety requirements before any real-world use.")
