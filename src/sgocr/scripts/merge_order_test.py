"""Direct unit test for _order_component_nodes reading-order fix.

Tests the row-bucketed sort against synthetic bbox layouts that previously
scrambled when classified as 'mixed' orientation. No model calls required.

Usage:
    PYTHONPATH=sgocr/src .venv_local/bin/python -m sgocr.scripts.merge_order_test
"""
from __future__ import annotations

from ..full_pipeline_dev40 import _order_component_nodes


def _make_nodes(entries: list[tuple[str, list[float]]]) -> list[dict]:
    """entries: [(text, [x1, y1, x2, y2]), ...]"""
    return [
        {"node_id": f"n{i:02d}", "text": text, "bbox": bbox}
        for i, (text, bbox) in enumerate(entries)
    ]


def _ordered_texts(nodes: list[dict]) -> list[str]:
    return [str(n["text"]) for n in _order_component_nodes(nodes)]


CASES: list[tuple[str, list[tuple[str, list[float]]], list[str]]] = [
    # ── 1. pure horizontal row (single line, left-to-right) ─────────────────
    (
        "Horizontal row — single line",
        [
            ("DRIVEWAY",  [300, 100, 420, 130]),
            ("KEEP",      [160, 100, 280, 130]),
            ("ASHFIELD",  [10,  100, 150, 130]),
        ],
        ["ASHFIELD", "KEEP", "DRIVEWAY"],
    ),
    # ── 2. vertical stack (sign text stacked top-to-bottom) ──────────────────
    (
        "Vertical stack — sign words top-to-bottom",
        [
            # shuffled input order mimics what the pipeline used to produce
            ("KEEP",      [30, 200, 200, 230]),
            ("COUNCIL",   [30, 150, 200, 180]),
            ("CLEAR",     [30, 300, 200, 330]),
            ("ASHFIELD",  [30, 100, 200, 130]),
            ("DRIVEWAY",  [30, 250, 200, 280]),
        ],
        ["ASHFIELD", "COUNCIL", "KEEP", "DRIVEWAY", "CLEAR"],
    ),
    # ── 3. two-line block (multi-line text, e.g. store sign) ─────────────────
    (
        "Two-line text block",
        [
            ("OPEN",     [120, 110, 200, 140]),
            ("24",       [215, 110, 255, 140]),
            ("HOURS",    [260, 110, 360, 140]),
            ("WELCOME",  [10,  60,  140, 90]),
            ("TO",       [150, 60,  200, 90]),
            ("OUR",      [210, 60,  270, 90]),
            ("STORE",    [280, 60,  370, 90]),
        ],
        ["WELCOME", "TO", "OUR", "STORE", "OPEN", "24", "HOURS"],
    ),
    # ── 4. two columns (left column top-to-bottom, right column top-to-bottom) ─
    (
        "Two columns — left then right, each top-to-bottom",
        [
            ("B2",  [200, 150, 280, 180]),
            ("A1",  [10,  100, 90,  130]),
            ("A2",  [10,  150, 90,  180]),
            ("B1",  [200, 100, 280, 130]),
        ],
        # Row-then-x: row1 = A1, B1 (similar y=115); row2 = A2, B2 (similar y=165)
        ["A1", "B1", "A2", "B2"],
    ),
    # ── 5. singleton ─────────────────────────────────────────────────────────
    (
        "Singleton node",
        [("ONLY", [50, 50, 150, 80])],
        ["ONLY"],
    ),
    # ── 6. mixed: diagonal text on a sign, left-to-right within each row ────
    (
        "Staggered rows — each word slightly offset in y",
        [
            # Words share approximate rows despite slight y variation
            ("STREET",  [180,  52,  290,  78]),
            ("MAIN",    [10,   48,  170,  75]),
            ("EAST",    [300,  55,  400,  80]),
            ("NO.",     [10,   95,  60,   120]),
            ("123",     [65,   92,  120,  118]),
        ],
        ["MAIN", "STREET", "EAST", "NO.", "123"],
    ),
]


def main() -> None:
    print("=" * 60)
    print("  _order_component_nodes — reading-order unit tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, entries, expected in CASES:
        nodes = _make_nodes(entries)
        got = _ordered_texts(nodes)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\n[{status}] {name}")
        if not ok:
            print(f"  expected: {expected}")
            print(f"  got:      {got}")
        else:
            print(f"  order: {got}")

    print(f"\n{'─'*60}")
    print(f"  Results: {passed}/{passed+failed} passed", "✓" if failed == 0 else "✗")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
