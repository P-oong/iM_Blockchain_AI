"""Shared, conservative market-key cleaning; this is not an address geocoder."""

import re
import unicodedata
from collections import Counter, defaultdict


Group = tuple[str, str, str]
MISSING_VALUES = {"", "nan", "none", "null", "nat", "미상", "없음"}
AREA_PATTERN = re.compile(r"^[가-힣][가-힣0-9·.]*[동읍면가](?:[0-9]+가)?$")
BUILDING_PATTERN = re.compile(
    r"상가|아파트|맨션|타운|시설|건축물|관리동|본관|별관|기숙사|주차|지하|대학|의료원|"
    r"^제[0-9]+동$|^주[0-9]+동|^층|^지면$"
)


def normalize_text(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "").strip()


def area_is_valid(value: str) -> bool:
    area = normalize_text(value)
    return bool(AREA_PATTERN.fullmatch(area)) and not BUILDING_PATTERN.search(area)


def market_key(district: str, area: str, industry: str) -> Group | None:
    group = tuple(normalize_text(value) for value in (district, area, industry))
    if any(value.lower() in MISSING_VALUES for value in group) or not area_is_valid(group[1]):
        return None
    return group


def geography_columns(header: list[str], area_column: str = "auto") -> tuple[str, str | None]:
    """Prefer corrected administrative geography, never silently fall back per row."""
    if area_column == "auto":
        area_column = "행정동" if "행정동" in header else "읍면동"
    if area_column not in header:
        raise ValueError(f"지역 열이 없습니다: {area_column}")
    code_column = area_column + "코드"
    return area_column, code_column if code_column in header else None


def normalize_area_code(value: str) -> str:
    value = normalize_text(value)
    if value.endswith(".0"):
        value = value[:-2]
    if not re.fullmatch(r"[0-9]{10}", value):
        raise ValueError("지역코드는 10자리 숫자여야 합니다.")
    return value


def canonical_geography(counts: dict[str, Counter]) -> tuple[dict, list[dict]]:
    """Resolve code/name inconsistencies using >=95% agreement within this source.

    This is source-internal normalization, not verification against an official
    geographic register. Ambiguous codes are excluded instead of guessed.
    """
    mapping, corrections = {}, []
    for code, names in sorted(counts.items()):
        ordered = sorted(names.items(), key=lambda item: (-item[1], item[0]))
        best, frequency = ordered[0]
        total = sum(names.values())
        if frequency / total < 0.95:
            corrections.append({"code": code, "action": "exclude_ambiguous_code", "records": total})
            continue
        mapping[code] = best
        for original, count in ordered[1:]:
            corrections.append({"code": code, "action": "normalize_to_source_consensus", "from": list(original),
                                "to": list(best), "records": count, "consensus_share": frequency / total})
    reverse = defaultdict(list)
    for code, name in mapping.items():
        reverse[name].append(code)
    for name, codes in reverse.items():
        if len(codes) > 1:
            for code in codes:
                del mapping[code]
                corrections.append({"code": code, "action": "exclude_name_code_collision", "name": list(name)})
    return mapping, corrections
