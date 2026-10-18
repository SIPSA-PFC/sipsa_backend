"""
core/names.py
-------------
Neighborhood name matching across sources.

The same neighborhood is spelled differently in each file
('Neópolis' / 'Neopolis', 'Dix-sept Rosado' / 'Dixsept Rosado',
'Felipe Camarão' / 'Filipe Camarão'). Names are compared on a key without
accents, case, spaces or punctuation, plus explicit aliases for real
spelling variants.
"""

from __future__ import annotations

import re
import unicodedata

# key of a variant spelling -> key of the canonical spelling
ALIASES = {
    "filipecamarao": "felipecamarao",   # IBGE 2010 spelling
}


def name_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    key = re.sub(r"[^a-z0-9]", "", text.lower())
    return ALIASES.get(key, key)
