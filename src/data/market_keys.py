"""Shared, conservative market-key cleaning; this is not an address geocoder."""

import re
import unicodedata


Group = tuple[str, str, str]
MISSING_VALUES = {"", "nan", "none", "null", "nat", "미상", "없음"}
AREA_PATTERN = re.compile(r"^[가-힣][가-힣0-9·.]*[동읍면가](?:[0-9]+가)?$")
BUILDING_PATTERN = re.compile(
    r"상가|아파트|맨션|타운|시설|건축물|관리동|본관|별관|기숙사|주차|지하|대학|의료원|"
    r"^제[0-9]+동$|^주[0-9]+동|^[가나다라마바사아자차카타파하]동$|^층|^지면$"
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
