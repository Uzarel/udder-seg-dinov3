import configparser

from typing import Any, Dict

def _coerce(value: str) -> Any:
    """Best-effort cast from string to int/float/bool/None, else return original string."""
    v = value.strip()
    low = v.lower()
    if low in {"none", "null"}:
        return None
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    # int first (so "42" -> 42, not 42.0)
    try:
        return int(v)
    except ValueError:
        pass
    # then float (so "0.005" -> 0.005, "1.0" -> 1.0)
    try:
        return float(v)
    except ValueError:
        pass
    return v  # leave as string

def get_config(path: str = "config.ini") -> Dict[str, Dict[str, Any]]:
    """
    Read an INI file and return a nested dict:
      { "GENERAL": {...}, "MODEL": {...}, "DATA": {...}, ... }
    Values are auto-cast to int/float/bool/None when possible.
    """
    parser = configparser.ConfigParser()
    parser.optionxform = str # preserve key casing
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read configuration file: {path}")

    cfg: Dict[str, Dict[str, Any]] = {}

    # Add all sections
    for section in parser.sections():
        cfg[section] = {k: _coerce(v) for k, v in parser.items(section, raw=True)}

    return cfg
