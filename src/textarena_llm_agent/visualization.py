from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .tracing import TextArenaRunTracer


class TextArenaVisualizationServer:
    def __init__(self, tracer: TextArenaRunTracer, *, host: str = "127.0.0.1", port: int = 8765,
                 eval_root: str | Path | None = None,
                 evidence_graph_path: str | Path | None = None) -> None:
        self.tracer = tracer
        self.host = host
        self.port = port
        self.eval_root = Path(eval_root) if eval_root else Path("workspace/eval_runs")
        self.evidence_graph_path = (
            Path(evidence_graph_path) if evidence_graph_path
            else Path("workspace/textarena_memory/evidence_graph.sqlite")
        )
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        port = self.httpd.server_port if self.httpd is not None else self.port
        return f"http://{self.host}:{port}/"

    def start(self, *, open_browser: bool = False) -> str:
        self.httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        if open_browser:
            webbrowser.open(self.url)
        return self.url

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()

    def _handler(self):
        tracer = self.tracer
        eval_root = self.eval_root
        evidence_graph_path = self.evidence_graph_path

        class Handler(BaseHTTPRequestHandler):
            server_version = "TextArenaAgentVisualization/0.1"

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_html(DASHBOARD_HTML)
                elif parsed.path == "/api/events":
                    limit = int(parse_qs(parsed.query).get("limit", ["600"])[0])
                    self._send_json({"events": tracer.read_events(limit=limit)})
                elif parsed.path == "/api/state":
                    self._send_json(tracer.read_state())
                elif parsed.path == "/api/control":
                    self._send_json(tracer.read_control())
                elif parsed.path.startswith("/api/eval/"):
                    game = parsed.path.rsplit("/", 1)[-1]
                    self._send_json(_read_eval(eval_root, game))
                elif parsed.path == "/api/policy_versions":
                    self._send_json(_policy_versions(evidence_graph_path))
                elif parsed.path == "/api/cache":
                    limit = int(parse_qs(parsed.query).get("limit", ["500"])[0])
                    self._send_json(_cache_history(tracer, limit=limit))
                elif parsed.path == "/api/tool_lifecycle":
                    self._send_json(_tool_lifecycle(evidence_graph_path))
                elif parsed.path == "/api/skill_timeline":
                    game = parse_qs(parsed.query).get("game", [None])[0]
                    self._send_json(_skill_timeline(evidence_graph_path, game=game))
                elif parsed.path == "/api/nashconv":
                    game = parse_qs(parsed.query).get("game", [None])[0]
                    self._send_json(_nashconv(eval_root, game=game))
                else:
                    self.send_error(404, "Not found")

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/api/control":
                    self.send_error(404, "Not found")
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                try:
                    update = json.loads(body)
                except Exception:
                    update = {}
                control = tracer.read_control()
                action = str(update.get("action") or "")
                if action == "pause":
                    control["paused"] = True
                elif action == "resume":
                    control["paused"] = False
                    control["step_requested"] = False
                elif action == "step":
                    control["paused"] = True
                    control["step_requested"] = True
                elif action == "stop":
                    control["stop_requested"] = True
                elif isinstance(update.get("control"), dict):
                    control.update(update["control"])
                tracer.write_control(control)
                tracer.emit("control_update", {"control": control})
                self._send_json(control)

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send_json(self, data: Any) -> None:
                raw = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _send_html(self, html: str) -> None:
                raw = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        return Handler


def serve_trace(trace_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> TextArenaVisualizationServer:
    server = TextArenaVisualizationServer(TextArenaRunTracer(trace_dir), host=host, port=port)
    server.start(open_browser=open_browser)
    return server


def _read_eval(eval_root: Path, game: str) -> dict[str, Any]:
    gdir = eval_root / game
    out: dict[str, Any] = {"game": game}
    for name in ("trend.json", "elo.json", "exploitability.json"):
        path = gdir / name
        if path.exists():
            try:
                out[name.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    timeline = gdir / "skill_timeline.jsonl"
    if timeline.exists():
        rows = []
        for line in timeline.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        out["skill_timeline"] = rows
    return out


def _policy_versions(graph_path: Path) -> dict[str, Any]:
    if not graph_path.exists():
        return {"nodes": [], "edges": [], "note": f"graph not found: {graph_path}"}
    try:
        import sqlite3
        conn = sqlite3.connect(str(graph_path))
        try:
            cur = conn.execute(
                "SELECT id, name, parent, skill_set_hash, tool_set_hash, created_at, attrs_json "
                "FROM policy_version ORDER BY created_at ASC"
            )
            nodes: list[dict[str, Any]] = []
            edges: list[dict[str, str]] = []
            for r in cur.fetchall():
                try:
                    attrs = json.loads(r[6] or "{}")
                except Exception:
                    attrs = {}
                nodes.append({"id": r[0], "name": r[1], "parent": r[2],
                              "skill_set_hash": r[3], "tool_set_hash": r[4],
                              "created_at": r[5], "attrs": attrs})
                if r[2]:
                    edges.append({"src": r[2], "dst": r[0]})
        finally:
            conn.close()
        return {"nodes": nodes, "edges": edges}
    except Exception as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}


def _cache_history(tracer: TextArenaRunTracer, *, limit: int = 500) -> dict[str, Any]:
    path = tracer.decision_frames_path
    if not path.exists():
        return {"frames": [], "summary": {"avg_hit_ratio": 0.0, "total_prompt_tokens": 0,
                                            "total_cached_tokens": 0, "n": 0}}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, limit):]:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        rows.append({
            "id": obj.get("id"), "step": obj.get("step"),
            "policy_version": obj.get("policy_version"),
            "prompt_tokens": int(obj.get("prompt_tokens") or 0),
            "completion_tokens": int(obj.get("completion_tokens") or 0),
            "cached_tokens": int(obj.get("cached_tokens") or 0),
            "cache_hit_ratio": float(obj.get("cache_hit_ratio") or 0.0),
            "created_at": obj.get("created_at"),
        })
    total_p = sum(r["prompt_tokens"] for r in rows)
    total_c = sum(r["cached_tokens"] for r in rows)
    avg = (sum(r["cache_hit_ratio"] for r in rows) / len(rows)) if rows else 0.0
    return {
        "frames": rows,
        "summary": {
            "avg_hit_ratio": round(avg, 4),
            "total_prompt_tokens": total_p,
            "total_cached_tokens": total_c,
            "overall_hit_ratio": round(total_c / total_p, 4) if total_p else 0.0,
            "n": len(rows),
        },
    }


def _tool_lifecycle(graph_path: Path) -> dict[str, Any]:
    if not graph_path.exists():
        return {"versions": [], "synthesized": [], "counts_by_status": {},
                "note": f"graph not found: {graph_path}"}
    try:
        import sqlite3
        conn = sqlite3.connect(str(graph_path))
        try:
            cur = conn.execute(
                "SELECT tv.id, tv.tool_id, tv.version, tv.status, tv.policy_version, "
                "       tv.replay_score, tv.ab_score, tv.unit_tests_passed, tv.created_at, t.name "
                "FROM tool_version tv LEFT JOIN tool t ON t.id = tv.tool_id "
                "ORDER BY tv.created_at ASC"
            )
            versions = [{
                "id": r[0], "tool_id": r[1], "version": r[2], "status": r[3],
                "policy_version": r[4], "replay_score": r[5], "ab_score": r[6],
                "unit_tests_passed": bool(r[7]), "created_at": r[8], "tool_name": r[9],
            } for r in cur.fetchall()]
            cur2 = conn.execute(
                "SELECT id, tool_id, name, status, version, policy_version, "
                "       replay_score, ab_score, created_at FROM synthesized_tool "
                "ORDER BY created_at ASC"
            )
            synthesized = [{
                "id": r[0], "tool_id": r[1], "name": r[2], "status": r[3], "version": r[4],
                "policy_version": r[5], "replay_score": r[6], "ab_score": r[7],
                "created_at": r[8],
            } for r in cur2.fetchall()]
        finally:
            conn.close()
        counts: dict[str, int] = {}
        for v in versions:
            counts[v["status"]] = counts.get(v["status"], 0) + 1
        for s in synthesized:
            counts[s["status"]] = counts.get(s["status"], 0) + 1
        return {"versions": versions, "synthesized": synthesized, "counts_by_status": counts}
    except Exception as exc:
        return {"versions": [], "synthesized": [], "counts_by_status": {}, "error": str(exc)}


def _skill_timeline(graph_path: Path, *, game: str | None = None) -> dict[str, Any]:
    if not graph_path.exists():
        return {"versions": [], "counts_by_status": {},
                "note": f"graph not found: {graph_path}"}
    try:
        import sqlite3
        conn = sqlite3.connect(str(graph_path))
        try:
            q = ("SELECT sv.id, sv.skill_id, sv.version, sv.status, sv.policy_version, "
                 "       sv.replay_score, sv.ab_score, sv.created_by, sv.created_at, "
                 "       s.name, s.game_id FROM skill_version sv "
                 "JOIN skill s ON s.id = sv.skill_id")
            params: tuple[Any, ...] = ()
            if game:
                q += " WHERE s.game_id = ?"
                params = (game,)
            q += " ORDER BY sv.created_at ASC"
            cur = conn.execute(q, params)
            versions = [{
                "id": r[0], "skill_id": r[1], "version": r[2], "status": r[3],
                "policy_version": r[4], "replay_score": r[5], "ab_score": r[6],
                "created_by": r[7], "created_at": r[8], "skill_name": r[9],
                "game_id": r[10],
            } for r in cur.fetchall()]
        finally:
            conn.close()
        counts: dict[str, int] = {}
        for v in versions:
            counts[v["status"]] = counts.get(v["status"], 0) + 1
        return {"versions": versions, "counts_by_status": counts}
    except Exception as exc:
        return {"versions": [], "counts_by_status": {}, "error": str(exc)}


def _nashconv(eval_root: Path, *, game: str | None = None) -> dict[str, Any]:
    if game:
        games = [game]
    elif eval_root.exists():
        games = [p.name for p in eval_root.iterdir() if p.is_dir()]
    else:
        games = []
    out: dict[str, Any] = {"games": {}}
    for g in games:
        path = eval_root / g / "exploitability.json"
        if path.exists():
            try:
                out["games"][g] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return out


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TextArena Agent Console</title>
  <style>
    :root { --bg:#f7f7f4; --ink:#202124; --muted:#65676b; --line:#d9d9d2; --panel:#ffffff; --accent:#136f63; --warn:#b54708; --good:#18794e; }
    * { box-sizing: border-box; }
    body { margin:0; color:var(--ink); background:var(--bg); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:16px; background:#fff; }
    h1 { margin:0; font-size:22px; letter-spacing:0; }
    .subtitle { color:var(--muted); margin-top:3px; font-size:13px; }
    .controls { display:flex; flex-wrap:wrap; gap:8px; }
    button { border:1px solid var(--line); background:#fff; color:var(--ink); width:38px; height:34px; border-radius:6px; cursor:pointer; font-weight:700; }
    button:hover { border-color:var(--accent); }
    main { display:grid; grid-template-columns:minmax(360px,1fr) minmax(360px,.9fr); gap:14px; padding:14px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    .head { padding:12px 14px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; gap:10px; }
    .head h2 { margin:0; font-size:15px; }
    .small { color:var(--muted); font-size:12px; }
    .content { padding:12px 14px; }
    .grid { display:grid; gap:4px; max-width:760px; }
    .cell { min-height:42px; border:1px solid var(--line); border-radius:4px; padding:5px; font:12px/1.25 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#fafaf8; display:flex; align-items:center; justify-content:center; text-align:center; }
    .card { border:1px solid var(--line); border-radius:6px; padding:10px; margin-bottom:8px; background:#fff; }
    .card h3 { margin:0 0 6px; font-size:14px; }
    .score { color:var(--good); font-weight:800; }
    .risk { color:var(--warn); }
    .events { max-height:76vh; overflow:auto; padding:10px; }
    .event { border-left:3px solid var(--accent); background:#fff; border-radius:6px; padding:9px 10px; margin-bottom:8px; border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line); }
    .event-title { font-weight:700; display:flex; justify-content:space-between; gap:8px; }
    pre { white-space:pre-wrap; word-break:break-word; font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin:8px 0 0; max-height:260px; overflow:auto; }
    @media (max-width: 960px) { header { display:block; } .controls { margin-top:10px; } main { grid-template-columns:1fr; padding:8px; } }
    .eval-panel { margin:0 14px 14px; }
    .charts { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
    .chart-card { border:1px solid var(--line); border-radius:6px; padding:8px; background:#fff; }
    .chart-card h3 { margin:0 0 6px; font-size:13px; }
    select { border:1px solid var(--line); border-radius:6px; padding:4px 8px; }
    @media (max-width: 1100px) { .charts { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div><h1>TextArena Agent Console</h1><div class="subtitle">State, legal candidates, evaluator feedback, memory, and skill evolution history.</div></div>
    <div class="controls">
      <button title="Pause" onclick="control('pause')">||</button>
      <button title="Step" onclick="control('step')">>|</button>
      <button title="Resume" onclick="control('resume')">></button>
      <button title="Stop" onclick="control('stop')">X</button>
    </div>
  </header>
  <main>
    <section>
      <div class="head"><h2>Game</h2><span id="status" class="small">loading</span></div>
      <div class="content">
        <div id="board" class="grid"></div>
        <div id="decision"></div>
        <h2>Candidate Ranking</h2>
        <div id="candidates"></div>
      </div>
    </section>
    <section>
      <div class="head"><h2>Trace</h2><span id="control-state" class="small"></span></div>
      <div id="events" class="events"></div>
    </section>
  </main>
  <section class="eval-panel">
    <div class="head"><h2>Evolution &amp; Evaluation</h2>
      <select id="eval-game" onchange="loadEval()">
        <option value="TicTacToe">TicTacToe</option>
        <option value="KuhnPoker">KuhnPoker</option>
        <option value="SimpleNegotiation">SimpleNegotiation</option>
        <option value="Stratego">Stratego</option>
      </select>
    </div>
    <div class="content">
      <div class="charts">
        <div class="chart-card"><h3>Win-rate / Reward by phase (improvement curve)</h3><canvas id="trend-chart" width="420" height="180"></canvas></div>
        <div class="chart-card"><h3>Elo progression</h3><canvas id="elo-chart" width="420" height="180"></canvas></div>
        <div class="chart-card"><h3>Skill evolution timeline</h3><canvas id="skill-chart" width="420" height="180"></canvas></div>
      </div>
      <div id="exploitability-box" class="card"></div>
    </div>
  </section>
<script>
const state = { events: [] };
async function fetchJson(url, opts) { const r = await fetch(url, opts); return await r.json(); }
async function control(action) { await fetchJson('/api/control', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action})}); await refresh(); }
function latestEvent(name) { for (let i=state.events.length-1;i>=0;i--) if (state.events[i].event===name) return state.events[i]; return null; }
function escapeHtml(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function renderBoard(s) {
  const snap = s.snapshot || {};
  const board = snap.board || s.state?.visible_state?.board || null;
  const el = document.getElementById('board');
  if (Array.isArray(board) && Array.isArray(board[0])) {
    el.style.gridTemplateColumns = `repeat(${board[0].length}, minmax(34px, 1fr))`;
    el.innerHTML = board.flatMap(row => row.map(cell => `<div class="cell">${escapeHtml(cell || '.')}</div>`)).join('');
  } else { el.style.gridTemplateColumns = '1fr'; el.innerHTML = `<pre>${escapeHtml(JSON.stringify(s.state?.visible_state || {}, null, 2))}</pre>`; }
  document.getElementById('status').textContent = `${snap.env_id || '?'} | turn ${snap.turn ?? '?'} | player ${snap.current_player ?? '?'} | done ${!!snap.done}`;
}
function renderDecision() {
  const d = latestEvent('decision_resolved') || latestEvent('decision_selected');
  const e = latestEvent('evaluation_complete');
  const payload = d?.payload?.decision || {};
  document.getElementById('decision').innerHTML = `<h2>Decision</h2><div class="card">
    <h3>${escapeHtml(payload.candidate_id || 'no decision')} ${escapeHtml(payload.action_text || '')}</h3>
    <div><b>Rationale:</b> ${escapeHtml(payload.rationale || '')}</div>
    <div><b>Plan:</b> ${escapeHtml(payload.plan || '')}</div>
    <div><b>Evaluator:</b> ${escapeHtml(e?.payload?.evaluation?.critique || '')}</div>
  </div>`;
}
function renderCandidates() {
  const cands = latestEvent('candidates_ranked')?.payload?.candidates || [];
  document.getElementById('candidates').innerHTML = cands.slice(0, 10).map(c => `<div class="card">
    <h3>${escapeHtml(c.candidate_id)} <span class="score">${Number(c.score || 0).toFixed(2)}</span> ${escapeHtml(c.action_text || '')}</h3>
    <div class="small">${escapeHtml((c.reasons || []).join(' | '))}</div>
    <div class="risk">${escapeHtml((c.risks || []).join(' | '))}</div>
  </div>`).join('');
}
function renderEvents() {
  document.getElementById('events').innerHTML = state.events.slice(-80).reverse().map(e => `<div class="event">
    <div class="event-title"><span>${escapeHtml(e.event)}</span><span class="small">${escapeHtml(e.created_at || '')}</span></div>
    <pre>${escapeHtml(JSON.stringify(e.payload || {}, null, 2))}</pre>
  </div>`).join('');
}
async function refresh() {
  const [s, events, controlState] = await Promise.all([fetchJson('/api/state'), fetchJson('/api/events?limit=800'), fetchJson('/api/control')]);
  state.events = events.events || [];
  document.getElementById('control-state').textContent = `paused ${!!controlState.paused} | stop ${!!controlState.stop_requested}`;
  renderBoard(s); renderDecision(); renderCandidates(); renderEvents();
}
async function loadEval() {
  const game = document.getElementById('eval-game').value;
  const data = await fetchJson(`/api/eval/${game}`);
  renderTrend(data.trend); renderElo(data.elo, data); renderSkill(data.skill_timeline || []);
  const ex = data.exploitability || {};
  document.getElementById('exploitability-box').innerHTML = `<h3>Exploitability</h3><div><b>method:</b> ${escapeHtml(ex.method||'-')}</div>` +
    (ex.method === 'best_response_value' ? `<div><b>BR value:</b> ${ex.br_value} | game value: ${ex.game_value} | <b>exploitability:</b> ${ex.exploitability}</div>` : '') +
    (ex.method === 'loss_rate_vs_optimal' ? `<div><b>P(loss) vs optimal:</b> ${ex.p_loss} (${ex.losses}/${ex.games})</div>` : '') +
    (ex.note ? `<div class="small">${escapeHtml(ex.note)}</div>` : '');
}
function drawLineChart(canvas, series, opts={}) {
  const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height);
  const labels = opts.labels || series.map((_,i)=>i); const allVals = series.flatMap(s=>s.values);
  const max = Math.max(0.1, ...allVals), min = Math.min(0, ...allVals);
  const pad=24, w=canvas.width-pad-8, h=canvas.height-pad-8;
  const x = i => pad + (i/(Math.max(1,labels.length-1)))*w;
  const y = v => pad + h - ((v-min)/(max-min||1))*h;
  ctx.strokeStyle='#999'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,pad+h); ctx.lineTo(pad+w,pad+h); ctx.stroke();
  ctx.fillStyle='#65676b'; ctx.font='10px sans-serif';
  labels.forEach((l,i)=>{ if(i%2===0){ctx.fillText(l,x(i)-6,pad+h+12);} });
  const colors=['#136f63','#b54708','#18794e','#65676b'];
  series.forEach((s,si)=>{
    ctx.strokeStyle=colors[si%colors.length]; ctx.beginPath();
    s.values.forEach((v,i)=> i===0?ctx.moveTo(x(i),y(v)):ctx.lineTo(x(i),y(v)));
    ctx.stroke(); ctx.fillStyle=ctx.strokeStyle; ctx.fillText(s.label, pad+w-90, pad+10+si*14);
  });
}
function renderTrend(trend){ const c=document.getElementById('trend-chart'); if(!trend||!trend.length){c.getContext('2d').clearRect(0,0,c.width,c.height);return;}
  drawLineChart(c, [
    {label:'win_rate', values:trend.map(t=>t.win_rate)},
    {label:'avg_reward', values:trend.map(t=>t.avg_reward)},
    {label:'skills', values:trend.map(t=>t.skill_count)},
  ], {labels:trend.map(t=>t.bin)});
}
function renderElo(elo){ const c=document.getElementById('elo-chart'); if(!elo){c.getContext('2d').clearRect(0,0,c.width,c.height);return;}
  const hist=elo.history||[]; const names=Object.keys(elo.ratings||{});
  const series=names.map(n=>({label:n, values:hist.filter(h=>h.a===n||h.b===n).map(h=>h.a===n?h.rating_a:h.rating_b)}));
  if(series.every(s=>!s.values.length)){ series.forEach(s=>s.values=[elo.ratings[s.label]]); }
  drawLineChart(c, series, {labels:hist.map((_,i)=>i)});
}
function renderSkill(timeline){ const c=document.getElementById('skill-chart'); if(!timeline.length){c.getContext('2d').clearRect(0,0,c.width,c.height);return;}
  const byEvent={}; timeline.forEach(t=>{byEvent[t.event]=(byEvent[t.event]||0)+1;});
  const labels=Object.keys(byEvent); drawBarChart(c, labels, labels.map(l=>byEvent[l]));
}
function drawBarChart(canvas, labels, values){ const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height);
  const max=Math.max(1,...values); const pad=24,w=canvas.width-pad-8,h=canvas.height-pad-8,bw=w/Math.max(1,labels.length);
  ctx.strokeStyle='#999'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,pad+h); ctx.lineTo(pad+w,pad+h); ctx.stroke();
  const colors=['#136f63','#b54708','#18794e','#65676b','#8e44ad'];
  labels.forEach((l,i)=>{ ctx.fillStyle=colors[i%colors.length]; const v=values[i]; ctx.fillRect(pad+i*bw+2, pad+h-(v/max)*h, bw-4, (v/max)*h);
    ctx.fillStyle='#65676b'; ctx.font='10px sans-serif'; ctx.fillText(l.slice(0,10), pad+i*bw+2, pad+h+12); });
}
setInterval(refresh, 900); refresh(); loadEval();
</script>
</body>
</html>
"""
