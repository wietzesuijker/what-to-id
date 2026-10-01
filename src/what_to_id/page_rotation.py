"""Rotation design: one blind index.html where every identifier cycles through all lists.

Unlike the "sets" design (page.py), no browser is stuck on one list: each click on "Next batch"
advances a per-browser, per-group cycle through the blind list labels in a random permutation
drawn once and stored in localStorage. Every label gets the same share of every identifier's
effort. Each browser also starts every list and group at its own random batch and wraps around,
so identifiers working the same group spread over the served window instead of all opening the
same first batches (disjoint dealing). The embedded data maps opaque list labels (the same
letters as manifest.arm_labels) to per-group ordered lists of Identify URLs; nothing else. No
arm name, batch_id, card fact, or thumbnail ever reaches this page, so a batch from any list
renders the same: a group name, a batch count and one Identify link.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from what_to_id.batches import batch_key
from what_to_id.manifest import Manifest
from what_to_id.page import ARM_WORDS, group_name
from what_to_id.page_method import METHOD_CSS, METHOD_JS, method_section
from what_to_id.page_style import CSS, FONTS

_SG = '"Space Grotesk",Inter,system-ui,sans-serif'

ROTATION_CSS = f"""
[hidden]{{display:none!important}}
#groupList{{display:grid;grid-template-columns:repeat(auto-fill,minmax(10rem,1fr));gap:.5rem;
margin:.8rem 0}}
.gbtn{{display:flex;flex-direction:column;gap:.1rem;padding:.65rem .8rem;background:var(--card);
color:var(--cink);border:1px solid var(--line);border-radius:12px;font:inherit;text-align:left;
cursor:pointer;transition:border-color .12s}}
.gbtn:hover{{border-color:var(--acc)}}
.gbtn.is-current{{border-color:var(--acc);
box-shadow:0 0 0 2px var(--acc)}}
.gbtn.is-empty{{opacity:.6}}
.gbtn b{{font-size:1.1rem;color:var(--acc-ink);font-family:{_SG}}}
.gbtn .lat{{color:var(--cmut);font-size:.8rem}}
.gbtn .left{{color:var(--cmut);font-size:.88rem;margin-top:.3rem}}
#runner{{margin-top:1.2rem}}
.runhead{{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;
gap:.3rem 1.2rem;margin:0 0 .5rem}}
.rungroup{{font-family:{_SG};font-size:1.3rem;font-weight:700;margin:0;color:var(--ink)}}
.runcode{{font-family:{_SG};font-size:1rem;font-weight:700;color:var(--acc);margin:0 0 .4rem}}
.bar{{height:.4rem;background:var(--panel);border-radius:4px;overflow:hidden;margin:0 0 1rem}}
.bar span{{display:block;height:100%;width:0;background:var(--acc)}}
.nextbtn{{display:block;width:100%;padding:1.1rem;font-size:1.15rem;font-weight:700;
background:var(--gd);color:#062a12;border:0;border-radius:12px;cursor:pointer;
font-family:{_SG};transition:filter .12s}}
.nextbtn:hover{{filter:brightness(1.08)}}
.nextbtn:disabled{{opacity:.5;cursor:default;filter:none}}
.nextbtn:focus-visible{{outline:3px solid var(--ink);outline-offset:3px}}
.hint{{display:flex;flex-wrap:wrap;gap:.3rem 1rem;color:var(--mut);font-size:.88rem;
margin:.6rem 0 0}}
.hint a{{font-weight:600}}
kbd{{font:600 .8rem/1 {_SG};padding:.15rem .4rem;border:1px solid #2a3a4d;border-bottom-width:2px;
border-radius:5px;background:var(--panel);color:var(--ink)}}
@media(hover:none){{.keys{{display:none}}}}
#runner .crumbs{{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;align-items:center}}
.linkbtn{{display:inline-flex;align-items:center;min-height:2.75rem;background:none;border:0;
padding:.35rem .5rem;font:inherit;font-size:.9rem;color:var(--mut);cursor:pointer;
text-decoration:underline;text-underline-offset:3px;transition:color .12s}}
.linkbtn:hover:not(:disabled){{color:var(--acc)}}
.linkbtn:disabled{{opacity:.4;cursor:default}}
.subhint{{width:100%;margin:0;color:var(--mut);font-size:.88rem}}
.pickhead{{font:700 1rem/1.3 {_SG};margin:1.2rem 0 .5rem;color:var(--ink);text-transform:none;
letter-spacing:normal}}
.resumerow{{margin:1.2rem 0 0}}
.reopenrow{{margin:.5rem 0 0}}
.reopenrow a{{font-weight:600}}
.donebox,.undo{{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;
gap:.6rem;margin-top:1rem;padding:.7rem 1rem;background:var(--panel);border:1px solid #2a3a4d;
border-radius:12px;color:var(--ink)}}
.donebox{{border-color:var(--gd)}}
.pillbtn{{padding:.5rem .9rem;background:var(--card);color:var(--cink);border:0;border-radius:8px;
font:inherit;font-weight:600;cursor:pointer}}
.undo .linkbtn{{color:var(--acc);font-weight:700}}
.cycle .n{{font-family:{_SG};font-weight:700}}
.cycle i{{font-style:normal;color:var(--acc);font-size:1.2rem}}
.cycle{{display:flex;align-items:center;gap:.5rem;margin:.4rem 0 .8rem}}
.cycle .n{{width:2.2rem;height:2.2rem;display:grid;place-items:center;border-radius:50%;
background:var(--card);color:var(--acc-ink)}}
""".strip()

# nextBatch is the one pure function driving the rotation: given the per-browser state, the
# embedded label -> group -> urls data, and the chosen group, it returns the next url (or done)
# and a new state. It never mutates its inputs, so it can be unit-tested outside the browser.
# groupProgress and withLast are pure helpers for the progress line and "Open again" link; they
# count batches over all lists together, so they say nothing about any one list.
ROTATION_JS = """
function nextBatch(state, data, group) {
  var perm = state.perm, K = perm.length;
  var progress = state.progress || {};
  var prev = progress[group] || {step: 0, pointers: {}, served: 0};
  var pointers = Object.assign({}, prev.pointers);
  var attempt, label, idx, list, foundLabel = null, foundIdx = -1;
  for (attempt = 0; attempt < K; attempt++) {
    label = perm[(prev.step + attempt) % K];
    idx = pointers[label] || 0;
    list = (data[label] && data[label][group]) || [];
    if (idx < list.length) { foundLabel = label; foundIdx = idx; break; }
  }
  var next = Object.assign({}, progress);
  if (foundLabel === null) {
    next[group] = {step: prev.step + K, pointers: pointers, served: prev.served};
    return {
      state: Object.assign({}, state, {progress: next}),
      url: null, label: null, done: true, batchNumber: prev.served
    };
  }
  pointers[foundLabel] = foundIdx + 1;
  var urls = data[foundLabel][group];
  var offset = ((state.offsets || {})[foundLabel] || {})[group] || 0;
  var served = prev.served + 1;
  next[group] = {step: prev.step + attempt + 1, pointers: pointers, served: served};
  return {
    state: Object.assign({}, state, {progress: next}),
    url: urls[(offset + foundIdx) % urls.length],
    label: foundLabel,
    done: false,
    batchNumber: served
  };
}
function leftInGroup(data, state, group) {
  var pointers = ((state.progress || {})[group] || {}).pointers || {};
  var left = 0;
  Object.keys(data).forEach(function(label){
    var list = (data[label] && data[label][group]) || [];
    left += Math.max(0, list.length - (pointers[label] || 0));
  });
  return left;
}
function groupProgress(state, data, group) {
  var total = 0;
  Object.keys(data).forEach(function(label){
    total += ((data[label] && data[label][group]) || []).length;
  });
  var left = leftInGroup(data, state, group);
  return {opened: total - left, left: left, total: total};
}
function withLast(state, group, url) {
  var last = Object.assign({}, state.last || {});
  last[group] = url;
  return Object.assign({}, state, {last: last});
}
function startOver(state, group) {
  var progress = Object.assign({}, state.progress || {});
  delete progress[group];
  var last = Object.assign({}, state.last || {});
  delete last[group];
  return Object.assign({}, state, {progress: progress, last: last});
}
(function(){
  var SKEY = 'what-to-id-rotation';
  var GKEY = 'what-to-id-rotation-group';
  var labels = Object.keys(DATA);
  var groups = [];
  labels.forEach(function(l){Object.keys(DATA[l]).forEach(function(g){
    if (groups.indexOf(g) === -1) groups.push(g);
  });});
  function name(g) { return GROUP_NAMES[g] || g; }
  groups.sort(function(a, b){ return name(a).localeCompare(name(b)); });
  function shuffled(arr) {
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }
  var state = null, newBuildNotice = false;
  try { state = JSON.parse(localStorage.getItem(SKEY) || 'null'); } catch (e) {}
  if (!state || !Array.isArray(state.perm) || state.perm.length !== labels.length) {
    state = {perm: shuffled(labels), progress: {}};
  }
  // A new daily build re-cuts every list, so old pointers and offsets point at other batches.
  // Keep the browser's list cycle, start the batches afresh. Tell a returning identifier once,
  // only if they had progress to lose.
  if (state.build !== BUILD) {
    var hadProgress = Object.keys(state.progress || {}).length > 0;
    state = {perm: state.perm, progress: {}, build: BUILD};
    if (hadProgress) newBuildNotice = true;
  }
  if (!state.offsets) {
    state.offsets = {};
    labels.forEach(function(l){
      state.offsets[l] = {};
      Object.keys(DATA[l]).forEach(function(g){
        state.offsets[l][g] = Math.floor(Math.random() * DATA[l][g].length);
      });
    });
  }
  var group = null;
  try { group = localStorage.getItem(GKEY); } catch (e) {}
  function save() {
    try {
      localStorage.setItem(SKEY, JSON.stringify(state));
      if (group) localStorage.setItem(GKEY, group);
    } catch (e) {}
  }
  function $(id) { return document.getElementById(id); }
  var mapLink = $('mapLink'), mapBase = mapLink && mapLink.getAttribute('data-base');
  function setMap(g) {
    if (mapLink) mapLink.href = g ? mapBase + '#groups=' + encodeURIComponent(g) : mapBase;
  }
  var STALE_DAYS = 2;
  (function(){
    var t = $('picked');
    var m = t && /^(\\d{4})-(\\d{2})-(\\d{2})$/.exec(t.getAttribute('datetime') || '');
    if (!m) return;
    var now = new Date();
    var days = Math.floor((Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())
      - Date.UTC(+m[1], m[2] - 1, +m[3])) / 864e5);
    if (isNaN(days)) return;
    var before = t.previousSibling, after = t.nextSibling;
    if (days > STALE_DAYS && before && after) {
      before.nodeValue = 'These records were picked ';
      t.textContent = days + ' days ago';
      after.nodeValue = ', and some may have an ID by now.' + after.nodeValue.slice(1);
    } else {
      t.textContent = days <= 0 ? 'today' : days === 1 ? 'yesterday' : days + ' days ago';
    }
  })();
  var picker = $('picker'), runner = $('runner'), groupList = $('groupList');
  var runGroup = $('runGroup'), runnerCode = $('runnerCode'), bar = $('bar');
  var nextBtn = $('nextBtn'), reopen = $('reopen'), doneMsg = $('doneMsg'), hintBox = $('hintBox');
  var startOverBtn = $('startOver'), undoMsg = $('undoMsg'), undoText = $('undoText');
  var buildMsg = $('buildMsg'), resumeRow = $('resumeRow'), resumeBtn = $('resumeBtn');
  var undoState = null, undoTimer = null;
  if (newBuildNotice) buildMsg.hidden = false;
  $('buildMsgClose').addEventListener('click', function(){ buildMsg.hidden = true; });
  function hideUndo() { undoMsg.hidden = true; undoState = null; clearTimeout(undoTimer); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }
  function renderPicker() {
    groupList.innerHTML = '';
    groups.forEach(function(g){
      var left = leftInGroup(DATA, state, g);
      var m = /^(.*) \\((.*)\\)$/.exec(name(g));
      var b = el('button', 'gbtn' + (g === group ? ' is-current' : '') +
        (left === 0 ? ' is-empty' : ''));
      b.type = 'button';
      b.appendChild(el('b', '', m ? m[1] : name(g)));
      if (m) b.appendChild(el('span', 'lat', m[2]));
      b.appendChild(el('span', 'left', left === 0 ? 'All opened' : left + ' batches left'));
      if (g === group) b.appendChild(el('span', 'left', 'Last used'));
      b.addEventListener('click', function(){ group = g; save(); renderRunner(true); });
      groupList.appendChild(b);
    });
    var known = group && groups.indexOf(group) !== -1;
    resumeRow.hidden = !known;
    if (known) {
      var r = /^(.*) \\((.*)\\)$/.exec(name(group));
      resumeBtn.textContent = 'Continue with ' + (r ? r[1] : name(group));
    }
    setMap(null);
    picker.hidden = false;
    runner.hidden = true;
  }
  function renderRunner(focus) {
    picker.hidden = true;
    runner.hidden = false;
    setMap(group);
    var p = groupProgress(state, DATA, group);
    runGroup.textContent = name(group);
    runnerCode.textContent = p.opened === 0 ? p.total + ' batches to go' :
      p.opened + ' of ' + p.total + ' batches opened';
    bar.style.width = (p.total ? 100 * p.opened / p.total : 0) + '%';
    var url = (state.last || {})[group];
    reopen.hidden = !url || p.opened === 0;
    if (url) { reopen.href = url; reopen.textContent = 'Open batch ' + p.opened + ' again'; }
    var isDone = p.left === 0;
    doneMsg.hidden = !isDone;
    nextBtn.hidden = isDone;
    nextBtn.disabled = isDone;
    hintBox.hidden = isDone;
    startOverBtn.disabled = p.opened === 0;
    if (focus && !nextBtn.disabled) nextBtn.focus();
  }
  nextBtn.addEventListener('click', function(){
    hideUndo();
    var res = nextBatch(state, DATA, group);
    state = res.done ? res.state : withLast(res.state, group, res.url);
    if (!res.done) window.open(res.url, '_blank', 'noopener');
    save();
    renderRunner(false);
  });
  // N opens the next batch. Held keys do not repeat, so one press is one batch.
  document.addEventListener('keydown', function(ev){
    if ((ev.key !== 'n' && ev.key !== 'N') || ev.repeat) return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (/^(INPUT|TEXTAREA|SELECT)$/.test((ev.target || {}).tagName || '')) return;
    if (runner.hidden || nextBtn.hidden || nextBtn.disabled) return;
    ev.preventDefault();
    nextBtn.click();
  });
  function toPicker() {
    hideUndo();
    renderPicker();
    var cur = groupList.querySelector('.gbtn.is-current');
    if (cur) cur.focus();
  }
  $('changeGroup').addEventListener('click', toPicker);
  $('pickOther').addEventListener('click', toPicker);
  // Start over puts this group's batches back, keeps the list cycle, and offers Undo for 10 s.
  startOverBtn.addEventListener('click', function(){
    undoState = state;
    state = startOver(state, group);
    save();
    renderRunner(false);
    undoText.textContent = 'Started over. All ' + leftInGroup(DATA, state, group) +
      ' batches in ' + name(group) + ' are back.';
    undoMsg.hidden = false;
    clearTimeout(undoTimer);
    undoTimer = setTimeout(hideUndo, 10000);
  });
  $('undoBtn').addEventListener('click', function(){
    if (undoState) { state = undoState; save(); renderRunner(false); }
    hideUndo();
  });
  // Every visit opens on the picker, so a new visitor or the next person on a shared computer
  // sees every group; a returning identifier resumes with one tap.
  resumeBtn.addEventListener('click', function(){ renderRunner(true); });
  renderPicker();
})();
""".strip()


def _rotation_data(manifest: Manifest) -> dict[str, dict[str, list[str]]]:
    """label -> group -> ordered Identify URLs, in the order batches were cut."""
    data: dict[str, dict[str, list[str]]] = {}
    for _bid, b in sorted(manifest.batches.items(), key=lambda kv: batch_key(kv[0])):
        label = manifest.arm_labels[b["arm"]]
        data.setdefault(label, {}).setdefault(str(b["group"]), []).append(b["url"])
    return data


def _runner(batch_size: int) -> str:
    """The batch runner. Identical for every list: nothing in it depends on the list served."""
    return (
        '<section id="runner" hidden>'
        '<div class="runhead"><h2 class="rungroup" id="runGroup"></h2>'
        '<p class="crumbs"><button class="linkbtn" id="changeGroup" type="button">'
        "Change group</button>"
        '<button class="linkbtn" id="startOver" type="button" disabled'
        ' aria-describedby="startOverHint">'
        "&#8634; Start over</button>"
        '<span class="subhint" id="startOverHint">Start over puts this group\'s batches back.'
        "</span></p></div>"
        '<p class="runcode" id="runnerCode" aria-live="polite">Batch 0</p>'
        '<div class="bar" aria-hidden="true"><span id="bar"></span></div>'
        '<button class="nextbtn" id="nextBtn" type="button" aria-keyshortcuts="n">'
        "Next batch</button>"
        f'<p class="hint" id="hintBox"><span>Opens up to {batch_size} records in a new '
        "iNaturalist tab. ID what you can, then come back.</span>"
        "<span>Sign in to iNaturalist first. Some records may no longer need an ID, so a batch "
        f"can show fewer than {batch_size}.</span>"
        "<span>Your place is saved in this browser only.</span>"
        '<span class="keys">Press <kbd>N</kbd> for the next batch.</span></p>'
        '<p class="reopenrow"><a id="reopen" href="#" target="_blank" rel="noopener" hidden>'
        "Open this batch again</a></p>"
        '<p class="donebox" id="doneMsg" hidden>'
        "<span>You opened every batch in this group. To go back to one, use Open again or "
        "Start over.</span>"
        '<button class="pillbtn" id="pickOther" type="button">Pick another group</button></p>'
        '<p class="undo" id="undoMsg" aria-live="polite" hidden><span id="undoText"></span>'
        '<button class="linkbtn" id="undoBtn" type="button">Undo</button></p>'
        "</section>\n"
    )


def render_rotation_index(manifest: Manifest, *, title: str, map_href: str | None = None) -> str:
    """One blind page: pick a group, then cycle "Next batch" through the lists in equal share."""
    data = _rotation_data(manifest)
    groups = sorted({g for by_group in data.values() for g in by_group})
    group_names = {g: group_name(g) for g in groups}
    sub = (
        "Records that needed an ID in British Columbia, picked "
        f'<time id="picked" datetime="{escape(manifest.freeze)}" title="{escape(manifest.freeze)}">'
        f"{escape(manifest.freeze)}</time>."
        if manifest.freeze
        else ""
    )
    if map_href:
        sub += (
            f' <a id="mapLink" href="{escape(map_href)}" data-base="{escape(map_href)}">'
            "See today's records on a map</a>."
        )
    sub_html = f'<p class="sub">{sub.strip()}</p>\n' if sub else ""
    build_msg = (
        '<p class="undo" id="buildMsg" aria-live="polite" hidden>'
        "<span>New batches today. Your count starts again.</span>"
        '<button class="linkbtn" id="buildMsgClose" type="button">Got it</button></p>\n'
    )
    body = (
        f"{build_msg}"
        '<section id="picker"><p class="resumerow" id="resumeRow" hidden>'
        '<button class="nextbtn" id="resumeBtn" type="button">Continue</button></p>'
        '<h2 class="pickhead">Pick a group</h2>'
        '<p class="hint">Choose a group you know. Open a batch in iNaturalist, '
        "ID what you can, then return.</p>"
        '<div id="groupList"></div></section>\n'
        f"{_runner(int(manifest.batch_size))}"
        f"{method_section(manifest, len(data))}"
    )
    script = (
        f"var BUILD={json.dumps(manifest.created_at)};"
        f"var DATA={json.dumps(data, sort_keys=True)};"
        f"var GROUP_NAMES={json.dumps(group_names, sort_keys=True)};"
        f"{ROTATION_JS}"
        f"{METHOD_JS}"
    )
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{title}</title>\n{FONTS}<style>{CSS}{ROTATION_CSS}{METHOD_CSS}</style>\n</head>\n<body>\n"
        "<noscript><p>This page needs JavaScript to deal batches.</p></noscript>\n"
        f"<main>\n<h1>{title}</h1>\n{sub_html}{body}"
        f"</main>\n<script>{script}</script>\n</body>\n</html>\n"
    )


def write_rotation_site(
    out_dir: Path | str, manifest: Manifest, *, map_href: str | None = None
) -> list[Path]:
    """Write a single blind index.html for the rotation design."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html = render_rotation_index(manifest, title="What to ID next in BC", map_href=map_href)
    low = html.lower()
    for w in ARM_WORDS:
        if w in low:
            raise ValueError(f"index.html: arm name {w!r} leaked into the rotation page")
    path = out / "index.html"
    path.write_text(html)
    return [path]
