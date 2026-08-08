"""Skill Alignment: data-driven canonical label adapter for OG output.

Hypothesis (Skill Alignment): 专用检测模型(LLMDet)输出与 Brain 期望的 label 分布
不一致(train-test mismatch)。Qwen 微调模型学到了 ALFRED canonical 词汇(如
CellPhone/KeyChain/CD/SoapBar)，而 LLMDet/Qwen-generic 输出 soft 形式
(phone/key/cd/soap bar)。canonicalize_label 将任何输入 label 对齐到 EB/ALFRED
canonical vocabulary。

设计原则:
  - 数据驱动: 索引来自 EB 的 canonical vocabulary (alfred_objs), 非硬编码规则。
  - 保守: 仅在 soft_type 能匹配到唯一 canonical 或 observed_objects 时改写;
    否则保留原 label (不引入错误映射)。
  - 家族保护: teapot→kettle 这类近类映射只在 canonical 无更好候选时使用。
"""
from __future__ import annotations

import os
import re
import string
from typing import Dict, List, Optional, Sequence, Tuple

# ---- canonical vocabulary (from EmbodiedBench eb_alfred utils.py) ----
# alfred_objs: all objects in EB action space (find/pick/toggle/etc.)
ALFRED_OBJS: List[str] = [
    'Cart', 'Potato', 'Faucet', 'Ottoman', 'CoffeeMachine', 'Candle', 'CD', 'Pan', 'Watch',
    'HandTowel', 'SprayBottle', 'BaseballBat', 'CellPhone', 'Kettle', 'Mug', 'StoveBurner', 'Bowl',
    'Toilet', 'DiningTable', 'Spoon', 'TissueBox', 'Shelf', 'Apple', 'TennisRacket', 'SoapBar',
    'Cloth', 'Plunger', 'FloorLamp', 'ToiletPaperHanger', 'CoffeeTable', 'Spatula', 'Plate', 'Bed',
    'Glassbottle', 'Knife', 'Tomato', 'ButterKnife', 'Dresser', 'Microwave', 'CounterTop',
    'GarbageCan', 'WateringCan', 'Vase', 'ArmChair', 'Safe', 'KeyChain', 'Pot', 'Pen', 'Cabinet',
    'Desk', 'Newspaper', 'Drawer', 'Sofa', 'Bread', 'Book', 'Lettuce', 'CreditCard', 'AlarmClock',
    'ToiletPaper', 'SideTable', 'Fork', 'Box', 'Egg', 'DeskLamp', 'Ladle', 'WineBottle', 'Pencil',
    'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'SaltShaker', 'PepperShaker',
    'Pillow', 'Bathtub', 'SoapBottle', 'Statue', 'Fridge', 'Sink',
]

# ALFRED object set (AW uses ALFRED object names, mostly same as EB with some differences)
ALFRED_OBJS_AW: List[str] = ALFRED_OBJS + ['AppleSliced', 'LettuceSliced', 'TomatoSliced',
    'PotatoSliced', 'BreadSliced', 'Toaster', 'StoveKnob', 'BathtubBasin', 'SinkBasin',
    'CoffeeMachine', 'Faucet', 'Microwave', 'StoveBurner', 'GarbageCan', 'Shelf', 'CounterTop',
    'Cabinet', 'Drawer', 'Safe', 'DeskLamp', 'FloorLamp', 'Bed', 'Desk', 'Dresser', 'Sofa',
    'SideTable', 'DiningTable', 'CoffeeTable', 'ArmChair', 'Ottoman', 'Cart', 'Bathtub',
    'Toilet', 'ToiletPaperHanger', 'LaundryHamper', 'Box', 'Pot', 'Pan', 'Plate', 'Bowl',
    'Mug', 'Cup', 'Glassbottle', 'WineBottle', 'WateringCan', 'SprayBottle', 'SoapBottle',
    'DishSponge', 'HandTowel', 'Cloth', 'SoapBar', 'Pillow', 'Pen', 'Pencil', 'Watch',
    'AlarmClock', 'Book', 'Newspaper', 'Laptop', 'CellPhone', 'RemoteControl', 'CD', 'CreditCard',
    'Statue', 'Vase', 'Candle', 'TennisRacket', 'BaseballBat', 'BasketBall', 'KeyChain',
    'ButterKnife', 'Knife', 'Fork', 'Spoon', 'Ladle', 'Spatula', 'ToiletPaper', 'TissueBox',
    'Egg', 'Apple', 'Potato', 'Tomato', 'Lettuce', 'Bread', 'Kettle', 'WateringCan',
    'PepperShaker', 'SaltShaker', 'Plunger', 'Candle',
]

# EB-only pickable objects (canonical target for OG when in EB)
_PICKABLE_OBJS: List[str] = [
    'KeyChain', 'Potato', 'Pot', 'Pen', 'Candle', 'CD', 'Pan', 'Watch', 'Newspaper', 'HandTowel',
    'SprayBottle', 'BaseballBat', 'Bread', 'CellPhone', 'Book', 'Lettuce', 'CreditCard', 'Mug',
    'AlarmClock', 'Kettle', 'ToiletPaper', 'Bowl', 'Fork', 'Box', 'Egg', 'Spoon', 'TissueBox',
    'Apple', 'TennisRacket', 'Ladle', 'WineBottle', 'Cloth', 'Plunger', 'SoapBar', 'Pencil',
    'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'Spatula', 'SaltShaker',
    'Plate', 'PepperShaker', 'Pillow', 'Glassbottle', 'SoapBottle', 'Knife', 'Statue', 'Tomato',
    'ButterKnife', 'WateringCan', 'Vase',
]


def pascal_to_soft(name: str) -> str:
    """'CellPhone' -> 'cell phone'; 'CD' -> 'cd'."""
    n = str(name).strip()
    if n == 'CD':
        return 'cd'
    n = re.sub(r"(\w)([A-Z])", r"\1 \2", n)
    return n.lower()


def soft_to_pascal(name: str) -> str:
    """'cell phone' -> 'CellPhone'; 'cd' -> 'CD'."""
    w = str(name).strip().lower()
    if w == 'cd':
        return 'CD'
    return ''.join(string.capwords(x) for x in w.split())


def _build_index(objects: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Return (soft→canonical, compact→[canonical]) indexes."""
    soft2canon: Dict[str, str] = {}
    compact2canon: Dict[str, List[str]] = {}
    for obj in objects:
        soft = pascal_to_soft(obj)
        soft2canon[soft] = obj
        compact = re.sub(r"[^a-z0-9]", "", soft)
        compact2canon.setdefault(compact, []).append(obj)
    return soft2canon, compact2canon


_SOFT2CANON, _COMPACT2CANON = _build_index(ALFRED_OBJS)
_SOFT2CANON_AW, _COMPACT2CANON_AW = _build_index(ALFRED_OBJS_AW)


def _strip_ids(name: Optional[str]) -> str:
    if not name:
        return ""
    t = re.sub(r"\s+\d+$", "", str(name).strip())
    return t


def _strip_sliced(name: str) -> str:
    """'sliced tomato' / 'TomatoSliced' / 'tomato sliced' / 'a sliced tomato' → 'tomato'."""
    if not name:
        return name
    t = str(name).strip()
    # strip article first: "a sliced tomato" / "the sliced tomato" -> "sliced tomato"
    t = re.sub(r"^\s*(a|an|the)\s+", "", t, flags=re.I)
    # leading "sliced X" or trailing "X Sliced" / "XSliced"
    t2 = re.sub(r"(?i)^sliced\s+", "", t)
    t2 = re.sub(r"(?i)\s+sliced$", "", t2)
    t2 = re.sub(r"(?i)^(apple|lettuce|tomato|potato|bread)sliced$", r"\1", t2)
    return t2 or name


def canonicalize_label(
    label: str,
    query: str = "",
    last_goto: Optional[str] = None,
    observed_objects: Optional[Sequence[str]] = None,
    env_name: str = "alfworld",
) -> str:
    """Align `label` to the canonical (EB/ALFRED) vocabulary.

    Priority:
      1. observed_objects instance match (best)
      2. soft_type match to canonical vocabulary
      3. compact (no-space, lowercase) match to canonical vocabulary
      4. keep original (conservative; never invent)
    """
    if not label:
        return label or ""
    lab = str(label).strip()
    # already canonical (in vocabulary)? For EB, *Sliced variants are NOT canonical
    # (EB pick vocab is the base class); strip them to base.
    if lab in ALFRED_OBJS:
        return lab
    if env_name != "eb-alfred" and lab in ALFRED_OBJS_AW:
        return lab

    index_soft = _SOFT2CANON_AW if env_name == "alfworld" else _SOFT2CANON
    index_compact = _COMPACT2CANON_AW if env_name == "alfworld" else _COMPACT2CANON

    # 1. observed_objects instance match: find canonical type among observed
    if observed_objects:
        lab_soft = pascal_to_soft(_strip_ids(lab))
        lab_compact = re.sub(r"[^a-z0-9]", "", lab_soft)
        for obs in observed_objects:
            obs_type = _strip_ids(str(obs))
            obs_compact = re.sub(r"[^a-z0-9]", "", pascal_to_soft(obs_type))
            if obs_compact == lab_compact:
                return str(obs)  # exact instance (e.g. "CellPhone 1")

    # 2. soft match
    lab_soft = pascal_to_soft(_strip_ids(lab))
    if lab_soft in index_soft:
        return index_soft[lab_soft]

    # 3. compact match (unique)
    lab_compact = re.sub(r"[^a-z0-9]", "", lab_soft)
    if lab_compact in index_compact and len(index_compact[lab_compact]) == 1:
        return index_compact[lab_compact][0]

    # 3c. containment match (unique prefix/contained): phone→cellphone,
    #     key→keychain, glass→glassbottle, remote→remotecontrol
    _contain = [c for c in index_compact if lab_compact in c and len(c) - len(lab_compact) >= 0]
    if len(_contain) == 1:
        canon_list = index_compact[_contain[0]]
        if len(canon_list) == 1:
            return canon_list[0]
        # prefer the one whose soft contains query family exactly
        for cand in canon_list:
            if lab_compact == re.sub(r"[^a-z0-9]", "", pascal_to_soft(cand)):
                return cand
        return canon_list[0]

    # 3d. near-class canonical (query not in EB vocab → map to EB object)
    if query:
        nc = near_class_canonical(query)
        if nc:
            return nc

    # 3e. sliced/state-suffix strip: "sliced tomato"/"TomatoSliced" → Tomato
    #     (EB pick vocab is the base class; AD must not emit *Sliced actions)
    _base = _strip_sliced(lab_soft)
    if _base != lab_soft:
        base_canon = canonicalize_label(_base, query=query, last_goto=last_goto,
                                        observed_objects=observed_objects,
                                        env_name=env_name)
        return base_canon

    # 3b. AW sliced forms
    if env_name == "alfworld":
        m = re.match(r"^(apple|lettuce|tomato|potato|bread)sliced$", lab_compact, re.I)
        if m:
            return m.group(1).capitalize() + "Sliced"

    # 4. query family protection: if query's soft matches a canonical, use it
    if query:
        q_compact = re.sub(r"[^a-z0-9]", "", pascal_to_soft(_strip_ids(query)))
        if q_compact in index_compact and len(index_compact[q_compact]) == 1:
            return index_compact[q_compact][0]

    # 5. conservative: keep original
    return label


# ---- Query-family drift detection (teapot→kettle, keys→key) ----
def query_family_ok(query: str, label: str) -> bool:
    """True if label is in the same 'family' as query (not a harmful drift).

    Compared on CANONICAL form, so 'wooden table'↔'dining table' both canonicalize
    to DiningTable and are considered same-family (correct). A real drift like
    'teapot'→'kettle' where canonical differs and families differ → False.
    """
    if not label:
        return False
    q = _strip_ids(query).lower()
    l = _strip_ids(label).lower()
    q_compact = re.sub(r"[^a-z0-9]", "", pascal_to_soft(q))
    l_compact = re.sub(r"[^a-z0-9]", "", pascal_to_soft(l))
    if not q_compact or not l_compact:
        return True
    # canonicalize both; if they land on the same canonical object → same family
    q_canon = canonicalize_label(label=q, query=query, env_name="eb-alfred")
    l_canon = canonicalize_label(label=label, query=query, env_name="eb-alfred")
    q_canon_soft = re.sub(r"[^a-z0-9]", "", pascal_to_soft(_strip_ids(q_canon)))
    l_canon_soft = re.sub(r"[^a-z0-9]", "", pascal_to_soft(_strip_ids(l_canon)))
    if q_canon_soft and l_canon_soft and q_canon_soft == l_canon_soft:
        return True
    # also allow subword containment as family
    if q_compact in l_compact or l_compact in q_compact:
        return True
    # near-class family (e.g. teapot→Kettle is same family for EB)
    nc_q = near_class_canonical(query)
    nc_l = near_class_canonical(label)
    if nc_q and nc_l and nc_q == nc_l:
        return True
    return False


# ---- Known safe near-class map (only for EB objects) ----
# e.g. teapot (not in EB vocab) → Kettle (EB has Kettle). phone→CellPhone handled by
# canonical vocabulary (CellPhone). key→KeyChain handled by vocabulary (KeyChain).
_NEAR_CLASS: Dict[str, str] = {
    "teapot": "Kettle",
    "coffeepot": "Kettle",
    "phone": "CellPhone",
    "cellphone": "CellPhone",
    "cell": "CellPhone",
    "key": "KeyChain",
    "keys": "KeyChain",
    "keychain": "KeyChain",
    "setofkeys": "KeyChain",
    "trashcan": "GarbageCan",
    "garbagecan": "GarbageCan",
    "couch": "Sofa",
    "fridge": "Fridge",
    "refrigerator": "Fridge",
    "remote": "RemoteControl",
    "remotecontrol": "RemoteControl",
    "toiletpaperroll": "ToiletPaper",
    "sink": "Sink",
    "glass": "Glassbottle",
    "glassbottle": "Glassbottle",
    "winebottle": "WineBottle",
    "spraybottle": "SprayBottle",
    "soapbottle": "SoapBottle",
    "soapbar": "SoapBar",
    "handtowel": "HandTowel",
    "dishsponge": "DishSponge",
    "wateringcan": "WateringCan",
    "butterknife": "ButterKnife",
    "basketball": "BasketBall",
    "baseballbat": "BaseballBat",
    "tennisracket": "TennisRacket",
    "toiletpaper": "ToiletPaper",
    "creditcard": "CreditCard",
    "alarmclock": "AlarmClock",
    "remotecontrol": "RemoteControl",
    "countertop": "CounterTop",
    "diningtable": "DiningTable",
    "sidetable": "SideTable",
    "coffeetable": "CoffeeTable",
    "toiletpaperhanger": "ToiletPaperHanger",
    "floorlamp": "FloorLamp",
    "desklamp": "DeskLamp",
    "coffeemachine": "CoffeeMachine",
    "stoveburner": "StoveBurner",
    "sinkbasin": "SinkBasin",
    "kitchentable": "DiningTable",
    "kitchen table": "DiningTable",
    "woodentable": "DiningTable",
    "wooden table": "DiningTable",
    "coffeetable": "CoffeeTable",
    "woodencoffeetable": "CoffeeTable",
    "kitchenisland": "CounterTop",
    "kitchen island": "CounterTop",
    "island": "CounterTop",
    "counter": "CounterTop",
    "metalrack": "Shelf",
    "metal rack": "Shelf",
    "table": "DiningTable",
    "coffeetable": "CoffeeTable",
    "sidetable": "SideTable",
    "endtable": "SideTable",
    "nightstand": "SideTable",
    "dresser": "Dresser",
    "cabinet": "Cabinet",
    "cupboard": "Cabinet",
    "wardrobe": "Cabinet",
    "bedsidecabinet": "Cabinet",
    "bathroomcabinet": "Cabinet",
    "armchair": "ArmChair",
    "armchair": "ArmChair",
    "chair": "ArmChair",
    "sofacouch": "Sofa",
    "garbage": "GarbageCan",
    "trash": "GarbageCan",
    "garbagebin": "GarbageCan",
    "laundryhamper": "LaundryHamper",
    "bathtubbasin": "BathtubBasin",
}


def near_class_canonical(query: str) -> Optional[str]:
    """Return canonical object for a query whose family has a near-class canonical."""
    if not query:
        return None
    q = re.sub(r"[^a-z0-9]", "", _strip_ids(query).lower())
    # exact compact match
    if q in _NEAR_CLASS:
        return _NEAR_CLASS[q]
    # containment: query contains a known canonical keyword → map by the keyword.
    # Only a conservative allowlist of high-confidence multi-word patterns.
    for kw, canon in _CONTAIN_KEYWORD.items():
        if kw in q:
            return canon
    return None


# containment keywords: high-confidence multi-word furniture/tool patterns only.
# Applied AFTER exact near-class; conservative to avoid wrong mappings.
# Order matters: more specific (longer) patterns first.
_CONTAIN_KEYWORD = {
    "microwaveoventable": "CounterTop",
    "microwaveovencounter": "CounterTop",
    "kitchencounter": "CounterTop",
    "countertop": "CounterTop",
    "tvstand": "Dresser",       # tv stand → Dresser (EB has Dresser, matches v3 behavior)
    "tvremote": "RemoteControl",  # tv remote → RemoteControl
    "remotecontrol": "RemoteControl",
    "woodentable": "DiningTable",
    "coffeetable": "CoffeeTable",
    "sidetable": "SideTable",
    "diningtable": "DiningTable",
    "kitchentable": "DiningTable",
    "barofsoap": "SoapBar",       # bar of soap → SoapBar
    "soapbar": "SoapBar",
    "table": "DiningTable",       # wooden table, dining room table, kitchen table (last)
    "desk": "Desk",
}
