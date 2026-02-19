"""
Technology ID Mapping
---------------------
Implements Table mapping.
"""

TECH_MAP = {
    "hadoop": 0,
    "ai": 1,
    "ml": 2,
    "cc": 3,
    "python": 4,
}


def map_technology(text: str) -> int:
    """
    Detect dominant technology keyword.

    Returns
    -------
    int
        Technology ID or -1 if none found.
    """

    text_lower = text.lower()

    for tech, idx in TECH_MAP.items():
        if tech in text_lower:
            return idx

    return -1
