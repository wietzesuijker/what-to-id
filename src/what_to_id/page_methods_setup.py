"""Methods sections 1 to 5: design, records, random split, the orders, serving batches.

Every number printed here is either this build's arm-blind aggregate (a total, or the smallest
and largest value over lists) or comes from running the real code on made-up input.
"""

from __future__ import annotations

import math
from collections import Counter

from what_to_id.manifest import Manifest
from what_to_id.page import group_name
from what_to_id.page_doc import Doc, a, codes, fold, see
from what_to_id.page_figures import (
    CHANCE_SIMS,
    chance_outside,
    deal_figure,
    flow_figure,
    list_share_figure,
    serving_figure,
)
from what_to_id.page_order_figures import BS, ORDER_FIGURES
from what_to_id.page_order_text import (
    ORDER_CAPTION,
    ORDER_DETAIL,
    ORDER_PARAMS,
    ORDER_TEXT,
    ORDER_WHY,
    order_name,
    primary_text,
)
from what_to_id.page_svg import fmt, range_words
from what_to_id.rest import REST_AFTER, REST_DAYS, REST_FRONT

_SETUP = "docs/protocol.md#proposed-set-up-for-bc"


def _span(lo: int, hi: int) -> str:
    return fmt(lo) if lo == hi else f"{fmt(lo)} to {fmt(hi)}"


def build_facts(m: Manifest) -> dict:
    """Arm-blind aggregates of this build: totals, and smallest and largest over lists."""
    per_list = dict.fromkeys(m.arms, 0)
    by_group: dict[str, dict[str, list[int]]] = {}
    for b in m.batches.values():
        per_list[b["arm"]] += len(b["ids"])
        by_group.setdefault(str(b["group"]), {}).setdefault(b["arm"], []).append(len(b["ids"]))
    groups, n_per_list = [], Counter()
    for g in sorted(by_group, key=group_name):
        lists = by_group[g]
        recs = [sum(lists.get(x, [])) for x in m.arms]
        nb = [len(lists.get(x, [])) for x in m.arms]
        sizes = [s for x in m.arms for s in lists.get(x, [])]
        n_per_list.update(nb)
        groups.append(
            {
                "name": group_name(g),
                "records": sum(recs),
                "recs": (min(recs), max(recs)),
                "list_recs": sorted(recs),
                "batches": (min(nb), max(nb)),
                "sizes": (min(sizes), max(sizes)),
            }
        )
    counts = list(per_list.values())
    b_max = max((g["batches"][1] for g in groups), default=0)
    return {
        "b_max": b_max,
        "cap_reached": bool(m.max_batches) and b_max >= int(m.max_batches),
        "k": len(m.arms),
        "pool_rows": int(m.pool_rows),
        "served_rows": sum(counts),
        "n_batches": len(m.batches),
        "batch_size": int(m.batch_size),
        "max_batches": m.max_batches,
        "list_min": min(counts),
        "list_max": max(counts),
        "groups": groups,
        "modal_batches": max(n_per_list, key=lambda n: (n_per_list[n], n)) if n_per_list else 0,
    }


def design_section(m: Manifest, f: dict, doc: Doc) -> str:
    primary = primary_text(m.arms)
    rows = [
        [
            "Question",
            "Does the order of records change how many get an ID, and which? More in "
            f"{a('README.md#what-the-lists-test', 'the README')}.",
        ],
        [
            "Randomised",
            f"Each record, to one of {f['k']} lists ({see('methods-split', 'Section 3')}).",
        ],
        [
            "Compared",
            "Each participant with themself: their count on each list against their "
            f"count on the control ({see('methods-analysis', 'Section 7')}).",
        ],
        [
            "Lists differ in",
            "The order: how records are sorted and cut into batches, and so, under a cap on "
            f"batches, which records are served ({see('methods-orders', 'Section 4')}).",
        ],
        [
            "Control",
            '<span id="identify-default">Newest first. Identify shows records newest first by '
            f"default (iNaturalist source {doc.cite('inatsource')}), so identifiers who do not "
            "use this page likely meet the newest records first, which is the control's "
            "order.</span>",
        ],
        [
            "Effect measured",
            "Meeting records in one list's batches in place of newest-first batches, served with "
            f"random starts ({see('methods-estimand', 'Section 5')}).",
        ],
        ["Primary outcome", f"{primary} ({see('methods-outcomes', 'Section 6')})."],
        [
            "Test",
            "Record-level re-randomisation test: one-sided for the confirmatory lists, "
            "Holm-adjusted over them, and two-sided, not adjusted, for the exploratory one "
            f"({see('methods-analysis', 'Section 7')}).",
        ],
        [
            "Dates",
            f"Records added up to {m.freeze}. The build was given {m.d1} as the blitz start. "
            "The draft protocol proposes four weeks in November 2026 and leaves the dates open.",
        ],
        ["Analysis plan", "A draft for the BC team. It is not registered."],
    ]
    premise = (
        "<p>Identification is a rate-limiting step in getting records that are already observed "
        "to Research Grade, and so to GBIF. The Blitz the Gap authors call for expert "
        "identification blitzes and for more informative prioritisation of records to identify "
        f"{doc.cite('hebert2026')}. Expert identification blitzes have been run and studied "
        f"before {doc.cite('mesaglio2025')}. This test asks a narrower question: whether the "
        "order of records changes what gets identified.</p>"
        f'<p id="methods-draft">This build has {f["k"]} lists. '
        "Trial dates and analysis settings are provisional until the BC team finalises "
        f"{a(_SETUP, 'the protocol')}.</p>"
    )
    table = doc.table(
        "The design in one table.", ["Part", "This test"], rows, wrap=True, name="design"
    )
    fig = doc.figure(
        flow_figure(f),
        "Flow of records through this build, drawn in the style of a CONSORT flow diagram "
        f"{doc.cite('schulz2010')}. The counts are this build's. The lists are not named.",
    )
    return f"{premise}{table}{fig}" + codes("manifest", "assign", "batches", "page", "analysis")


def records_section(m: Manifest, f: dict, doc: Doc) -> str:
    sizes = sorted({g["records"] for g in f["groups"]}) or [0]
    if len(sizes) == 1 and sizes[0] > 0 and sizes[0] % 200 == 0:
        frame = (
            "<p><b>This build is a preview sample.</b> It is not the blitz build and not a random "
            "sample of BC records.</p>"
        )
        why = (
            f"<p>Every taxon group has exactly {fmt(sizes[0])} served records, a whole number of "
            "pages of 200, so its pull most likely stopped after a page cap. So it holds the "
            "newest records, by record number, of each taxon group added up to the freeze "
            "date.</p>"
        )
    else:
        frame = (
            "<p>This page cannot tell from the build record whether this pull stopped early. "
            "The blitz's daily job pulls with no page cap.</p>"
        )
        why = ""
    asks = (
        "in British Columbia (iNaturalist place 7085)",
        "need an ID (quality grade Needs ID)",
        "have a photo",
        "were observed on or after the pull's <code>d1</code> date. The daily job uses 2025-01-01.",
        f"were added to iNaturalist up to the freeze date, {m.freeze}",
    )
    detail = (
        "<p>The pull asks the iNaturalist API, one taxon group at a time, for records that are:"
        f"</p><ul>{''.join(f'<li>{s}</li>' for s in asks)}</ul>"
        "<p>It asks for the highest record numbers first, which are the newest records, 200 per "
        "page, and it can stop after a set number of pages. It then drops a record with no "
        "coordinates or no photo, and a record number that comes twice. The build record also "
        f"has a field named <code>d1</code>, but it holds another date: the blitz start given to "
        f"the build, {m.d1}. The build record does not keep the pull's <code>d1</code> or page "
        f"cap.</p>{why}"
        "<p>A taxon group is one of iNaturalist's iconic taxa, for example Birds (Aves). Each "
        "taxon group gets its own batches, and you pick one on this page.</p>"
    )
    return (
        "<p>The pull takes the BC records in iNaturalist that need an ID and have a photo, one "
        f"taxon group at a time. This build holds {fmt(f['pool_rows'])} records and serves "
        f"{fmt(f['served_rows'])} of them in {fmt(f['n_batches'])} batches.</p>"
        f"<p>This page is one build, of records added up to {m.freeze}, and it does not change "
        "by itself. In the blitz, a daily job adds new records, drops served records that no "
        f"longer need an ID, moves to the back of its list for {REST_DAYS} days a record "
        f"served in the first {REST_FRONT} batches on {REST_AFTER} builds in a row that still "
        "needs one, and builds again each day "
        f"({a('README.md#how-it-works', 'the README')}). The daily script defaults to all four "
        "lists with rotation. The image-based lists require embeddings prepared separately; "
        "the daily build checks the supplied files before using them.</p>"
        f"{frame}"
        + fold("what the pull asks for, and what the build record keeps", detail)
        + codes("inat", "pool_state", "daily")
    )


def split_section(m: Manifest, f: dict, doc: Doc) -> str:
    k = f["k"]
    same = (
        f"<p>Each record goes to one of {k} lists at random, so on average every list holds the "
        "same mix of taxon groups and observers. The lists differ in their order and so, under a "
        "cap on batches, in which records are served. "
    )
    if m.assignment == "keyed":
        lead = same + (
            "This build is keyed: a keyed hash of the record number (HMAC-SHA256 with a private "
            "key) picks the list, and a record stays on its list in every daily build.</p>"
        )
        how = (
            "<p>The list is the first 8 bytes of the hash, read as a number, modulo "
            f"{k}. It depends only on the record number and the key, not on the rest of the pool, "
            "so it does not change as the pool grows and shrinks. The key is kept private and, by "
            "the draft protocol, opened only after the read-back.</p>"
        )
    else:
        lead = same + (
            "This build is seeded: inside "
            "each stratum of taxon group and observer bucket, a seed shuffles the records, and "
            "the records are dealt to the lists in turn.</p>"
        )
        how = (
            "<p>This build split the records inside strata. A stratum is one taxon group and one "
            "observer bucket. An observer bucket is one of 3 groups made from a checksum (CRC-32) "
            "of the observer's user number, or a fourth bucket for records with no user. The "
            "buckets keep the mix of observers alike across lists. Each "
            "stratum's deal starts one list later than the one before, so left-over records do "
            "not all go to the same list. Inside a stratum, list sizes differ by one record at "
            "most.</p>"
        ) + doc.figure(
            deal_figure(k),
            "The split, run by the assignment code on 20 made-up bird records from two observer "
            f"buckets, with {k} made-up lists. Each square is a record in its shuffled place, "
            "coloured by the list it is dealt to. The second stratum starts one list later. The "
            f"numbers 1 to {k} name made-up lists, not the lists on this page.",
        )
    rows = [
        [
            g["name"],
            fmt(g["records"]),
            fmt(g["recs"][0]),
            fmt(g["recs"][1]),
            fmt(g["recs"][1] - g["recs"][0]),
        ]
        for g in f["groups"]
    ]
    table = doc.table(
        "Served records per taxon group in this build. The lists are not named, so the table "
        "gives only the smallest and the largest list.",
        ["Taxon group", "Records", "Smallest list", "Largest list", "Difference"],
        rows,
        name="sizes",
    )
    band = m.assignment == "keyed" and not f["cap_reached"]
    fig = doc.figure(
        list_share_figure(f["groups"], k, band),
        "The smallest and the largest list in each taxon group, as a share of that group's "
        "served records. The line joins the two. The upright line marks the even share, "
        f"{100 / k:.4g} percent."
        + (
            f" The band comes from {fmt(CHANCE_SIMS)} simulated fair splits of the same number of "
            "records over the same number of lists: it runs from the 2.5th percentile of the "
            "smallest list to the 97.5th percentile of the largest, so a line that reaches past "
            "it is wider than chance usually gives. See the "
            f'<a href="#methods-threats">threats</a> section.'
            if band
            else ""
        ),
        name="shares",
    )
    extra = _chance_text(f, doc) if m.assignment == "keyed" else ""
    return (
        lead
        + table
        + fold(
            "how the split is drawn, and how much list sizes differ by chance", how + fig + extra
        )
        + codes("assign")
    )


def _chance_text(f: dict, doc: Doc) -> str:
    """How much list sizes differ by chance under the keyed split, and how many lists fall
    outside the chance band against how many a fair split puts there."""
    k, gs = f["k"], f["groups"]
    outs = [chance_outside(g["list_recs"], k) for g in gs]
    seen, expect = sum(o[0] for o in outs), sum(o[1] for o in outs)
    fall = "falls" if seen == 1 else "fall"
    big = max(gs, key=lambda g: g["recs"][1] - g["recs"][0])
    small = min(gs, key=lambda g: g["recs"][1] - g["recs"][0])
    sd = math.sqrt(big["records"] * (1 / k) * (1 - 1 / k))
    return (
        "<p>The keyed split draws each record's list on its own. It does not deal a fixed number "
        "of records to each list, so list sizes in a taxon group differ by chance. For a group of "
        f"{fmt(big['records'])} records over {k} lists, one list's size has a standard deviation "
        f"of about {sd:.0f} records. In this build the largest difference between the smallest "
        f"and the largest list is {fmt(big['recs'][1] - big['recs'][0])} records in "
        f"{big['name'].split(' (')[0]}, and the smallest is "
        f"{fmt(small['recs'][1] - small['recs'][0])} records in "
        f"{small['name'].split(' (')[0]}. Over all {len(gs)} taxon groups, {seen} of the "
        f"{k * len(gs)} lists {fall} outside their group's band in {doc.ref('shares')}. A fair "
        f"split puts {expect:.1f} lists outside on average, over the same {fmt(CHANCE_SIMS)} "
        "simulated splits. See the row on chance differences between lists in the "
        '<a href="#methods-threats">threats</a> section.</p>'
    )


def cap_text(f: dict) -> str:
    """Whether this build's cap on batches decides which records are served, from its data."""
    cap = f["max_batches"]
    if not cap:
        return "This build has no cap on batches, so every record is served."
    if not f["cap_reached"]:
        return (
            f"This build has a cap of {cap} batches per list and taxon group. No list reaches "
            "it, so every record is served."
        )
    left = f["pool_rows"] - f["served_rows"]
    return (
        f"This build serves only the first {cap} batches of each list in each taxon group, so "
        f"the order also decides which records are served. {fmt(left)} records are not served."
    )


def orders_section(m: Manifest, f: dict, doc: Doc) -> str:
    cap = " " + cap_text(f)
    parts = [f"<p>This build has {f['k']} lists, each with its own order:</p>"]
    batch_detail = (
        "<p>Each list sorts its records one taxon group at a time, then cuts the sorted records "
        f"into batches of up to {f['batch_size']} records, in that order.{cap} The figures run "
        f"the order code on made-up records and cut the result into batches of up to {BS}.</p>"
    )
    for i, arm in enumerate(m.arms, 1):
        name = order_name(arm)
        fig = doc.figure(ORDER_FIGURES[arm](), f"{name}, {ORDER_CAPTION[arm]}")
        rule = (
            f'<dl class="rule"><dt>Rule</dt><dd>{ORDER_DETAIL[arm]}</dd>'
            f"<dt>Parameters</dt><dd>{ORDER_PARAMS[arm](doc)}</dd></dl>"
        )
        parts.append(
            f'<details class="order" id="order-{i}"><summary>{ORDER_TEXT[arm]}</summary>'
            f'{fig}<dl class="rule">'
            f"<dt>Why</dt><dd>{ORDER_WHY[arm](doc)}</dd></dl>" + rule + "</details>"
        )
    keys = ["arms", "cells"] if "gap_first" in m.arms or "similarity" in m.arms else ["arms"]
    if "similarity" in m.arms or "novelty" in m.arms:
        keys.append("embed")
    return "".join(parts) + fold("batch sizes and examples", batch_detail) + codes(*keys, "batches")


def serving_section(m: Manifest, f: dict, doc: Doc) -> str:
    k, n = f["k"], f["modal_batches"]
    lo, hi = min((g["batches"][0] for g in f["groups"]), default=0), f["b_max"]
    fig = doc.figure(
        serving_figure(k, n),
        "Serving, with made-up identifiers. Turn order: each press of Next batch takes the next "
        f"list in the browser's turn order, here with {k} lists. The two browsers drew different "
        "turn orders. Start batch: inside one list and taxon group, each browser starts at a "
        "random batch and wraps around. In this build a list has "
        f"{range_words(lo, hi)} batches in a taxon group, most often {n}.",
        name="serving",
    )
    rows = [[g["name"], _span(*g["batches"]), _span(*g["sizes"])] for g in f["groups"]]
    batches = doc.table(
        "Batches per list in each taxon group, and records per batch, in this build. The lists "
        "are not named, so the table gives the range over lists.",
        ["Taxon group", "Batches per list", "Records per batch"],
        rows,
        name="batches",
    )
    s_lo = min(g["sizes"][0] for g in f["groups"])
    s_hi = max(g["sizes"][1] for g in f["groups"])
    batches += (
        f"<p>Across this build a batch holds {range_words(s_lo, s_hi)} records. A batch is the "
        f"next {f['batch_size']} records of the sorted list, and the last batch of a list in a "
        "taxon group takes the records left, so it is usually smaller. How many records you see "
        "in a batch also depends on how many of them still need an ID when you open it.</p>"
    )
    edges = (
        "<li>A list with no batches left in a taxon group: the press goes to the next list in "
        "the turn order.</li>"
        "<li>No list with batches left: the page says you opened every batch in the group.</li>"
        "<li>A second device or browser draws its own turn order and start batches. Each one "
        "is even on its own.</li>"
        "<li>A new daily build: your place in each taxon group resets and your start batches "
        "are drawn again. Your turn order stays.</li>"
        "<li>Start over puts one group's batches back. The turn order and start batches "
        "stay.</li>"
        "<li>Your place is saved in this browser only.</li>"
    )
    cycle = "".join(f'<span class="n">{i}</span>' for i in range(1, k + 1))
    lead = (
        f"<p>Each press of Next batch takes the next of {k} lists, in a turn order your browser "
        "draws at random once, so within one build your batches split evenly over the lists "
        f'while every list has batches.</p><div class="cycle" aria-hidden="true">{cycle}'
        f"<i>&#8634;</i></div><p>So a person who opens {k} or more batches in a taxon group "
        "within one build gets every list in turn, while every list has batches. This matters "
        f"because a few identifiers make most of the IDs {doc.cite('hebert2026')}: if each "
        "person kept one list, the list that drew the busiest identifier would win whatever its "
        "order. Inside each list and taxon group your "
        "browser starts at a random batch.</p>"
        "<p>The page shows no list name or letter on a batch, only its number. Your IDs carry "
        "your name and count toward Research Grade, and Research Grade records can go on to "
        "GBIF.</p>"
    )
    estimand = (
        '<h4 id="methods-estimand">What a list changes, as you see it</h4>'
        "<p>A list decides which records share a batch and the order of its batches after your "
        "random start. When the build serves only the first batches of each list, it also "
        f"decides which records are served ({see('methods-orders', 'Section 4')} says whether "
        "this build does). It does not decide which batch you get first, or the order inside a "
        "batch. So the test measures the effect of a list's batches, each met from a random "
        "start, against newest-first batches met the same way.</p>"
    )
    detail = (
        "<p>Each batch is one iNaturalist Identify link that lists its record numbers and the BC "
        "place, and nothing else. Identify hides records that no longer need an ID (checked in a "
        "browser on 2026-09-11), so a batch can show fewer records than it holds. The draft "
        "protocol says Identify also hides records you already reviewed. This was not "
        "checked.</p>"
        "<p>The turn order is the same for every taxon group, and each group keeps its own place "
        "in it. Inside each list and group, your browser goes on from its random start batch and "
        "wraps around to the first batch. The design borrows the within-person comparison of "
        f"interleaved search tests {doc.cite('chapelle2012')}. "
        f"{a('README.md#experiment-diagram', 'The README')} draws the turn order for two made-up "
        f"identifiers.</p>{batches}<h4>Edge cases</h4><ul>{edges}</ul>"
        "<p>The batch link carries no order, so Identify sets the order inside a batch, and you "
        "can skip. A person who opens all of a list's batches in one group meets all its records "
        "that still need an ID.</p>"
    )
    return (
        f"{lead}{fig}{estimand}"
        + fold("batch links, batch sizes and edge cases", detail, "more-batches")
        + codes("page", "batches")
    )


__all__ = [
    "ORDER_CAPTION",
    "ORDER_DETAIL",
    "ORDER_PARAMS",
    "ORDER_TEXT",
    "ORDER_WHY",
    "build_facts",
    "cap_text",
    "design_section",
    "order_name",
    "orders_section",
    "records_section",
    "serving_section",
    "split_section",
]
