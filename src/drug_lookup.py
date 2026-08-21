"""
drug_lookup.py

Looks up a drug's SMILES structure from its NAME, using PubChem's free
public API. This is the bridge between "text extracted from a photo"
and "the SMILES string your model actually needs."

PubChem's name search is fairly strict about spelling, so we clean up
the OCR text first (strip dosage numbers like "300mg", extra
whitespace, punctuation) before querying -- this meaningfully improves
match rates since box photos usually have the drug name mixed in with
dosage and other printed text.
"""

import re
import requests

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def clean_ocr_text(raw_text: str) -> str:
    """
    Strip common non-name noise from OCR output: dosage amounts
    (300mg, 500 MG), units, and stray punctuation. Keeps only
    letters/spaces, uppercased-then-titled for consistency.
    """
    # Remove dosage-like patterns: digits followed by mg/mcg/g/ml etc.
    text = re.sub(r"\d+\s*(mg|mcg|g|ml|iu)\b", "", raw_text, flags=re.IGNORECASE)
    # Remove any remaining digits and punctuation.
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


def search_drug_names(partial_name: str, limit: int = 8) -> list[str]:
    """
    Return a LIST of drug name suggestions matching a partial/in-progress
    search string -- for a live autocomplete dropdown as the user types
    (e.g. "para" -> ["Paracetamol", "Paracoumarin", ...]).

    Different from autocomplete_drug_name() above, which returns just
    the single best correction for a likely-misspelled full name.
    """
    if not partial_name or len(partial_name) < 2:
        # Querying PubChem on 1 character returns huge, unhelpful results.
        return []

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{partial_name}/json"
    try:
        response = requests.get(url, params={"limit": limit}, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        return data.get("dictionary_terms", {}).get("compound", [])
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[drug_lookup] Search failed for {partial_name!r}: {e}")
        return []


def autocomplete_drug_name(partial_name: str) -> str | None:
    """
    Use PubChem's autocomplete endpoint to find the closest real
    compound name to a possibly-misspelled input. This is what
    rescues OCR misreads like "Rispcrdal" -> "Risperdal" (a single
    misread character is enough to fail an exact name lookup, but
    autocomplete is built to tolerate near-misses like this).
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{partial_name}/json"
    try:
        response = requests.get(url, params={"limit": 3}, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        suggestions = data.get("dictionary_terms", {}).get("compound", [])
        return suggestions[0] if suggestions else None
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        print(f"[drug_lookup] Autocomplete failed for {partial_name!r}: {e}")
        return None


def name_to_smiles(drug_name: str) -> str | None:
    """
    Query PubChem for a drug's canonical SMILES by name.
    Returns None if no match is found (bad OCR read, misspelling,
    or a name PubChem doesn't recognize) rather than raising --
    callers should treat None as "needs a manual SMILES fallback."

    Tries an exact name match first (fast, cheap); if that fails,
    falls back to PubChem's autocomplete to correct likely OCR typos
    before giving up.
    """
    cleaned = clean_ocr_text(drug_name)
    if not cleaned:
        return None

    smiles = _lookup_exact(cleaned)
    if smiles:
        return smiles

    # Exact match failed -- likely an OCR typo. Try to correct it.
    corrected_name = autocomplete_drug_name(cleaned)
    if corrected_name and corrected_name.lower() != cleaned.lower():
        print(f"[drug_lookup] Exact match failed for {cleaned!r}, "
              f"trying autocomplete suggestion {corrected_name!r}")
        return _lookup_exact(corrected_name)

    return None


def _lookup_exact(name: str) -> str | None:
    """Single exact-name lookup against PubChem. Internal helper."""
    url = f"{PUBCHEM_BASE}/compound/name/{name}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"[drug_lookup] PubChem returned status {response.status_code} for {name!r}")
            return None
        data = response.json()
        properties = data["PropertyTable"]["Properties"][0]
        # PubChem's API returns the SMILES under "ConnectivitySMILES" in the
        # response even when "CanonicalSMILES" is requested in the URL --
        # check both so this keeps working regardless of which key shows up.
        return properties.get("ConnectivitySMILES") or properties.get("CanonicalSMILES")
    except requests.RequestException as e:
        print(f"[drug_lookup] Network/connection error looking up {name!r}: {type(e).__name__}: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"[drug_lookup] Could not parse PubChem response for {name!r}: {e}")
        return None


def extract_drug_name_from_ocr(ocr_lines: list[str]) -> str | None:
    """
    Given a list of text lines detected on a drug box (from OCR),
    pick the most likely candidate for the drug name and try to
    resolve it to a SMILES string, trying each line in turn until
    one succeeds (box photos often have the drug name on one line
    and dosage/manufacturer info on others).
    """
    # Try longer lines first -- drug names are rarely single short words,
    # while manufacturer codes/dosages often are.
    candidates = sorted(ocr_lines, key=len, reverse=True)
    for line in candidates:
        smiles = name_to_smiles(line)
        if smiles:
            return smiles
    return None


if __name__ == "__main__":
    # Quick manual test -- requires internet access to pubchem.ncbi.nlm.nih.gov
    test_cases = ["ASPIRIN 300mg", "Warfarin Sodium 5mg", "not a real drug xyz123"]
    for name in test_cases:
        result = name_to_smiles(name)
        print(f"{name!r} -> {result}")