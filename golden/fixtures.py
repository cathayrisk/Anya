# -*- coding: utf-8 -*-
"""測試用的合成文章。

刻意**不**把 kerykeion.net 的真實內容存進 repo——那是別人的東西，
而且我們已經決定不做站台鏡像。這裡改用合成 markdown 複製「觀察到的結構」，
用來測切片邏輯；真實頁面的煙霧測試留給 `--live`。

三種形狀來自實測抽樣（18 篇）：
  A 逐相位   `## The Trine (120°)` ×5，每段內含固定五個 ### 子段。主要星體用，最長。
  B 扁平原型 `## Archetypal Meaning` 等，無相位分段。小行星用，本來就短。
  C 逐接觸   `## Uranus Square Partner's Nodes`，標題自由但含相位字。交點類用。
"""

_PARA = ("This placement describes a sustained negotiation between the two principles. "
         "It is neither fortunate nor unfortunate on its own; what matters is how the "
         "tension is metabolised over time and which of the two voices is habitually "
         "given the last word in moments of pressure. ") * 6


def _aspect_section(name: str) -> str:
    return f"""## {name}

### Archetypal Meaning

{_PARA}

### Manifestations

{_PARA}

### Resources

- [Related reading](https://kerykeion.net/x)
- [More](https://kerykeion.net/y)

### Growth Edge

{_PARA}

### Integration

{_PARA}
"""


SHAPE_A = "# Sun-Saturn Aspects\n\nIntro paragraph about the pairing.\n\n" + "\n".join(
    _aspect_section(n) for n in (
        "The Conjunction (0°)", "The Sextile (60°)", "The Square (90°)",
        "The Trine (120°)", "The Opposition (180°)")
) + f"""
## Minor Aspects

{_PARA}

## Integration: Working With Sun-Saturn in Daily Life

{_PARA}

## Related Articles

- [a](https://kerykeion.net/a)
- [b](https://kerykeion.net/b)
"""

SHAPE_B = f"""# Natal Moon-Nessus Aspects: The Emotional Reckoning

Short intro.

## Archetypal Meaning

{_PARA}

## Typical Manifestations

{_PARA}

## Resources and Potentials

{_PARA}

## The Growth Edge

{_PARA}

## Working with Moon-Nessus Energy

{_PARA}
"""

SHAPE_C = "# Nodes-Uranus Synastry\n\nIntro.\n\n" + "\n".join(
    f"## {t}\n\n{_PARA}\n" for t in (
        "Uranus Conjunct Partner's North Node",
        "Uranus Conjunct Partner's South Node",
        "Uranus Square Partner's Nodes",
        "Uranus Trine/Sextile Partner's North Node",
    )
) + f"\n## Working With Nodes-Uranus Synastry\n\n{_PARA}\n"

# 病態案例：單段就爆預算 → 中介必須回 miss，而不是盲切
SHAPE_HUGE = "# Huge\n\n## The Trine (120°)\n\n" + (_PARA * 40)
# 病態案例：認不出章節
SHAPE_NO_HEADINGS = "# Flat\n\n" + (_PARA * 3)
SHAPE_404 = "404 Not Found\n\nThe page you requested does not exist."

CORPUS = {
    "natal-sun-saturn-aspects": SHAPE_A,
    "natal-moon-nessus-aspects": SHAPE_B,
    "synastry-nodes-uranus-aspects": SHAPE_C,
    "natal-sun-fourth-house": SHAPE_B,
    "foundation-saturn": SHAPE_B,
    "natal-chart-ruler-fifth-house": SHAPE_B,
    "__huge__": SHAPE_HUGE,
    "__flat__": SHAPE_NO_HEADINGS,
    "__404__": SHAPE_404,
}


def fake_fetcher(url: str) -> str:
    """把 Broker 的網路呼叫換成本地合成內容。"""
    slug = url.rsplit("/", 1)[-1]
    if slug not in CORPUS:
        raise RuntimeError(f"fixture 缺少 {slug}")
    return CORPUS[slug]
