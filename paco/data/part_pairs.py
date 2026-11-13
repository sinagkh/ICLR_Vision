"""Named part groups (A, B) per object class, matched against the PACO-LVIS category names.

resolve_target_pairs(categories) returns the part pairs found in the annotation file; for PACO-LVIS v1 these are
the five classes in lists/part_pair_map.json (bicycle, chair, laptop, car, bottle); the other entries have no
matching object category. The instance lists in lists/ keep objects of these classes on which both groups
are annotated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


@dataclass(frozen=True)
class TargetPartPair:
    object_key: str
    object_aliases: tuple[str, ...]
    part_a_group: str
    part_a_aliases: tuple[str, ...]
    part_b_group: str
    part_b_aliases: tuple[str, ...]


TARGETED_V1 = [
    TargetPartPair("bicycle", ("bicycle",), "wheel", ("wheel", "rim"), "handlebar_frame", ("handlebar", "frame", "tube", "fork")),
    TargetPartPair("chair", ("chair",), "seat", ("seat",), "back", ("back", "stile", "spindle")),
    TargetPartPair("laptop", ("laptop", "laptop_computer"), "screen", ("screen",), "keyboard_body", ("keyboard", "base_panel", "touchpad", "body")),
    TargetPartPair("airplane", ("airplane", "aeroplane", "airplane_aeroplane"), "wing", ("wing",), "tail_body", ("tail", "body", "fuselage")),
    TargetPartPair("car", ("car", "car_automobile", "automobile"), "wheel", ("wheel", "rim"), "window_body", ("window", "windowpane", "windshield", "body", "roof")),
    TargetPartPair("bottle", ("bottle",), "cap_neck", ("cap", "neck", "closure", "top", "spout"), "body", ("body", "inner_body", "shoulder", "base")),
    TargetPartPair("bus", ("bus",), "wheel", ("wheel", "rim"), "window_body", ("window", "windshield", "body")),
    TargetPartPair("truck", ("truck",), "wheel", ("wheel", "rim"), "window_body", ("window", "windshield", "body")),
    TargetPartPair("motorcycle", ("motorcycle", "motorbike"), "wheel", ("wheel", "rim"), "handlebar_body", ("handlebar", "body", "seat")),
]


def resolve_target_pairs(categories: list[dict[str, Any]], preset: str = "targeted_v1") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if preset != "targeted_v1":
        raise ValueError(f"Unsupported PACO target preset: {preset}")
    object_cats = [c for c in categories if c.get("supercategory") == "OBJECT"]
    part_cats = [c for c in categories if c.get("supercategory") == "PART" and ":" in str(c.get("name", ""))]
    resolved: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"preset": preset, "unmatched": [], "resolved": []}
    for spec in TARGETED_V1:
        obj = _find_object(object_cats, spec.object_aliases)
        if obj is None:
            diagnostics["unmatched"].append({"object_key": spec.object_key, "reason": "object_category_not_found"})
            continue
        obj_norm = norm_name(obj["name"])
        part_pool = [p for p in part_cats if norm_name(str(p["name"]).split(":", 1)[0]) == obj_norm]
        a_parts = _find_parts(part_pool, spec.part_a_aliases)
        b_parts = _find_parts(part_pool, spec.part_b_aliases)
        if not a_parts or not b_parts:
            diagnostics["unmatched"].append(
                {
                    "object_key": spec.object_key,
                    "object_name": obj["name"],
                    "reason": "part_group_missing",
                    "part_a_matches": [p["name"] for p in a_parts],
                    "part_b_matches": [p["name"] for p in b_parts],
                    "available_parts": [p["name"] for p in part_pool],
                }
            )
            continue
        item = {
            "object_key": spec.object_key,
            "object_category_id": int(obj["id"]),
            "object_category_name": obj["name"],
            "part_pair_id": f"{spec.object_key}:{spec.part_a_group}x{spec.part_b_group}",
            "part_a_group_name": spec.part_a_group,
            "part_a_part_ids": [int(p["id"]) for p in a_parts],
            "part_a_part_names": [p["name"] for p in a_parts],
            "part_b_group_name": spec.part_b_group,
            "part_b_part_ids": [int(p["id"]) for p in b_parts],
            "part_b_part_names": [p["name"] for p in b_parts],
        }
        resolved.append(item)
        diagnostics["resolved"].append(item)
    return resolved, diagnostics


def _find_object(categories: list[dict[str, Any]], aliases: tuple[str, ...]) -> dict[str, Any] | None:
    alias_norm = {norm_name(a) for a in aliases}
    for cat in categories:
        name = norm_name(str(cat.get("name", "")))
        if name in alias_norm or any(a and a in name for a in alias_norm):
            return cat
    return None


def _find_parts(part_pool: list[dict[str, Any]], aliases: tuple[str, ...]) -> list[dict[str, Any]]:
    alias_norm = [norm_name(a) for a in aliases]
    out = []
    for part in part_pool:
        part_name = norm_name(str(part["name"]).split(":", 1)[1])
        if any(alias and alias in part_name for alias in alias_norm):
            out.append(part)
    return out

