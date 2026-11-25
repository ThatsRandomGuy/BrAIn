"""Small deterministic extractors for explicit locations and quantity bounds.

These rules intentionally abstain on questions, negated locations and pronoun-only
subjects. They do not attempt unrestricted natural-language understanding.
"""
from __future__ import annotations
import re

_SMALL = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20)))
_TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(),range(20,100,10)))

def parse_number(text: str) -> str | None:
    text = text.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?",text):
        return text
    words = text.replace("-"," ").split()
    total = current = 0
    seen = False
    previous = ""
    for word in words:
        if word == "and":
            if previous != "hundred":
                return None
        elif word in _SMALL:
            if previous in _SMALL or (previous in _TENS and _SMALL[word] >= 10):
                return None
            current += _SMALL[word]
            seen = True
        elif word in _TENS:
            if previous in _SMALL or previous in _TENS:
                return None
            current += _TENS[word]
            seen = True
        elif word == "hundred":
            if previous not in _SMALL or not 1 <= current <= 9:
                return None
            current *= 100
        else:
            return None
        previous = word
    if not seen or previous == "and":
        return None
    return str(total+current)

def quantity_bounds(text: str):
    """Yield explicit bound text, preserving strict versus inclusive comparators."""
    if "?" in text:
        return
    pattern = r"\b(under|less than|below|at most|no more than)\s+([a-z0-9.\- ]+?)\s+(kilograms?|kg|grams?|pounds?|lbs?|minutes?|hours?|dollars?|liters?)\b"
    normalized = re.sub(
        r"(?<=\d)(?=(?:kilograms?|kg|grams?|pounds?|lbs?|minutes?|hours?|dollars?|liters?)\b)",
        " ", text.lower())
    for match in re.finditer(pattern,normalized):
        comparator, number_text, unit = match.groups()
        number = parse_number(number_text)
        if number is None:
            continue
        unit = {"kg":"kilograms","lb":"pounds","lbs":"pounds"}.get(unit,unit)
        family = "weight" if unit in {"kilogram","kilograms","gram","grams","pound","pounds"} else unit.rstrip("s")
        yield family, f"{comparator} {number} {unit}"

def explicit_locations(text: str):
    """Yield subject, preposition, full target, and explicit update flag."""
    if "?" in text:
        return
    lowered = text.lower()
    for clause in re.split(r"[.!?;]\s*|,\s*",lowered):
        clause=clause.strip()
        if not clause or re.match(r"^(where|what|which|is|are|can|could|do|does)\b",clause):
            continue
        clause=re.sub(r"^(?:(?:actually|correction|update|now)\s*:\s*|(?:actually|correction|update)\s+)", "",clause)
        match=re.fullmatch(
            r"(?:the |my |our )?([a-z0-9][a-z0-9 -]*?)\s+(?:is|are)\s+(?:(now|currently|still)\s+)?"
            r"(in|on|inside|under|above|behind|beside|at|next to)\s+(?:the |my |our |an? )?(.+)",clause)
        moved=False
        if not match:
            relocation=re.fullmatch(
                r"(?:(?:i|we)\s+(?:have\s+)?)?moved\s+(?:the |my |our )?([a-z0-9][a-z0-9 -]*?)\s+to\s+(?:the |my |our |an? )?(.+)",clause)
            if relocation:
                subject,target=relocation.groups()
                preposition="in"
                moved=True
            else:
                relocation=re.fullmatch(
                    r"(?:the |my |our )?([a-z0-9][a-z0-9 -]*?)\s+(?:has been|was)\s+moved\s+to\s+(?:the |my |our |an? )?(.+)",clause)
                if not relocation:
                    continue
                subject,target=relocation.groups()
                preposition="in"
                moved=True
        else:
            subject, qualifier, preposition, target=match.groups()
            moved=qualifier=="now"
        target=re.split(r"\s+(?:instead of|rather than|but|and then)\b",target)[0].strip()
        if preposition == "at" and target.startswith(("least ", "most ")):
            continue
        if preposition in {"under","above"} and parse_number(target.split()[0]) is not None:
            continue
        if subject in {"it","they","this","that","draft document"} or not target:
            continue
        if any(word in subject.split() for word in ("not","never")):
            continue
        yield subject.strip(),preposition,target,moved

def placement_constraints(text: str):
    if "?" in text:
        return
    pattern=r"(?:^|[.;]\s*|:\s*)(?:the |my |our )?([a-z0-9][a-z0-9 -]*?)\s+(?:must|should|needs to)\s+(?:stay|remain|be kept)\s+(in|inside|on)\s+(?:the |an? )?([^.;]+)"
    for match in re.finditer(pattern,text.lower()):
        subject,preposition,target=match.groups()
        yield subject.strip(),f"{subject.strip()} {preposition} {target.strip()}"
