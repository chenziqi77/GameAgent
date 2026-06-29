from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def event_payloads(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e.get("payload", {}) for e in events if e.get("event") == name]


def board_frames(events: list[dict[str, Any]], latest_state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload", {})
        snapshot = payload.get("snapshot") if isinstance(payload, dict) else None
        if isinstance(snapshot, dict) and snapshot.get("board") is not None:
            frames.append({"event": event.get("event"), "created_at": event.get("created_at"), "snapshot": snapshot})
    if latest_state and isinstance(latest_state.get("snapshot"), dict) and latest_state["snapshot"].get("board") is not None:
        frames.append({"event": "latest_state", "created_at": latest_state.get("updated_at"), "snapshot": latest_state["snapshot"]})
    return frames


def latest_decisions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in event_payloads(events, "decision_resolved"):
        decision = payload.get("decision", {})
        if isinstance(decision, dict):
            rows.append(decision)
    return rows


def load_eval(eval_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not eval_root.exists():
        return out
    for game_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        game: dict[str, Any] = {}
        for name in ("trend.json", "elo.json", "exploitability.json"):
            data = read_json(game_dir / name)
            if data is not None:
                game[name.removesuffix(".json")] = data
        timeline = read_jsonl(game_dir / "skill_timeline.jsonl")
        if timeline:
            game["skill_timeline"] = timeline
        if game:
            out[game_dir.name] = game
    return out


def artifact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def report_path(output_dir: Path, basename: str, suffix: str, *, timestamp: str, timestamped: bool) -> Path:
    name = f"{basename}_{timestamp}{suffix}" if timestamped else f"{basename}{suffix}"
    return output_dir / name


def write_svg_frames(frames: list[dict[str, Any]], output_dir: Path) -> list[str]:
    svg_dir = output_dir / "frames"
    svg_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for idx, frame in enumerate(frames):
        board = frame.get("snapshot", {}).get("board")
        if not (isinstance(board, list) and board and isinstance(board[0], list)):
            continue
        rows, cols = len(board), len(board[0])
        cell, pad, title_h = 68, 16, 34
        width, height = cols * cell + pad * 2, rows * cell + pad * 2 + title_h
        parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"]
        parts.append("<rect width='100%' height='100%' fill='#ffffff'/>")
        title = f"#{idx} {frame.get('event') or ''} turn={frame.get('snapshot', {}).get('turn', '')} player={frame.get('snapshot', {}).get('current_player', '')}"
        parts.append(f"<text x='{pad}' y='22' font-family='Arial' font-size='14' fill='#555'>{escape(title)}</text>")
        y0 = pad + title_h
        for r, row in enumerate(board):
            for c, raw in enumerate(row):
                x, y = pad + c * cell, y0 + r * cell
                val = str(raw or ".")
                fill = "#f8fafc"
                if val in {"X", "O"}:
                    fill = "#e8f2ff" if val == "X" else "#fff1e6"
                parts.append(f"<rect x='{x}' y='{y}' width='{cell-5}' height='{cell-5}' rx='7' fill='{fill}' stroke='#c7c7c7'/>")
                parts.append(f"<text x='{x + (cell-5)/2}' y='{y + (cell-5)/2 + 8}' text-anchor='middle' font-family='Arial' font-size='26' fill='#202124'>{escape(val)}</text>")
        parts.append("</svg>")
        name = f"frame_{idx:03d}.svg"
        (svg_dir / name).write_text("\n".join(parts), encoding="utf-8")
        names.append(f"frames/{name}")
    return names


def render_markdown(output_dir: Path, trace_dir: Path, decisions: list[dict[str, Any]], frames: list[dict[str, Any]], eval_data: dict[str, Any], *, timestamp: str, timestamped: bool = True) -> Path:
    lines = ["# TextArena Trace Report", "", f"Trace: `{trace_dir}`", "", "## Decision Timeline", ""]
    for idx, d in enumerate(decisions):
        lines.append(f"{idx}. `{d.get('candidate_id', '')}` `{d.get('action_text', '')}` confidence={d.get('confidence', '')}")
        rationale = str(d.get("rationale", "")).strip()
        if rationale:
            lines.append(f"   - rationale: {rationale}")
    lines += ["", "## Board Frames", "", f"Frames: {len(frames)}", ""]
    lines += ["## Evaluation Summary", "", "```json", json.dumps(eval_data, ensure_ascii=False, indent=2, default=str), "```", ""]
    path = report_path(output_dir, "trajectory", ".md", timestamp=timestamp, timestamped=timestamped)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def render_html(trace_dir: Path, output_dir: Path, eval_data: dict[str, Any], svg_names: list[str], *, timestamp: str, timestamped: bool = True) -> tuple[Path, Path]:
    events = read_jsonl(trace_dir / "events.jsonl")
    latest_state = read_json(trace_dir / "latest_state.json") or {}
    state_frame_rows = read_jsonl(trace_dir / "state_frames.jsonl")
    frames = []
    for row in state_frame_rows:
        snap = row.get("snapshot") if isinstance(row, dict) else None
        if isinstance(snap, dict) and snap.get("board") is not None:
            frames.append({"event": "state_frame", "created_at": row.get("updated_at"), "snapshot": snap})
    if not frames:
        frames = board_frames(events, latest_state)
    decisions = latest_decisions(events)
    candidates = event_payloads(events, "candidates_ranked")
    last_candidates = candidates[-1].get("candidates", []) if candidates else []
    memory_events = [e for e in events if str(e.get("event", "")).startswith("memory") or "skill" in str(e.get("event", ""))]
    md_path = render_markdown(output_dir, trace_dir, decisions, frames, eval_data, timestamp=timestamp, timestamped=timestamped)

    parts: list[str] = []
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>TextArena Local Report</title>")
    parts.append("<style>body{margin:0;font-family:Arial,sans-serif;background:#f6f7f8;color:#202124}header{padding:18px 24px;background:#fff;border-bottom:1px solid #ddd}main{padding:16px;display:grid;gap:16px}section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.card{border:1px solid #ddd;border-radius:7px;padding:10px;margin:8px 0;background:#fff}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #e5e5e5;padding:7px;text-align:left;vertical-align:top}pre{background:#f8f9fa;padding:10px;border-radius:6px;white-space:pre-wrap;max-height:360px;overflow:auto}.small{color:#666;font-size:13px}img{max-width:100%;border:1px solid #ddd;border-radius:8px;background:#fff}.player{display:grid;gap:10px;max-width:760px}.player img{width:100%;max-height:520px;object-fit:contain}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}button{border:1px solid #ccc;background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer}input[type=range]{min-width:240px;flex:1}</style></head><body>")
    parts.append(f"<header><h1>TextArena Local Report</h1><div class='small'>trace: {escape(str(trace_dir))}</div><div class='small'>generated_at_utc: {escape(timestamp)}</div><div class='small'>Copy this report directory to any device and open report.html.</div></header><main>")
    if svg_names:
        first = escape(svg_names[0])
        frame_json = json.dumps(svg_names, ensure_ascii=False)
        parts.append(f"<section><h2>Animated Replay</h2><div class='player'><img id='replay-frame' src='{first}' alt='replay frame'><div class='controls'><button onclick='toggleReplay()' id='play-btn'>Play</button><button onclick='stepFrame(-1)'>Prev</button><button onclick='stepFrame(1)'>Next</button><input id='frame-slider' type='range' min='0' max='{len(svg_names)-1}' value='0' oninput='setFrame(Number(this.value))'><span id='frame-label' class='small'>1 / {len(svg_names)}</span></div></div><script>const replayFrames={frame_json};let replayIndex=0;let replayTimer=null;function setFrame(i){{replayIndex=Math.max(0,Math.min(replayFrames.length-1,i));document.getElementById('replay-frame').src=replayFrames[replayIndex];document.getElementById('frame-slider').value=String(replayIndex);document.getElementById('frame-label').textContent=(replayIndex+1)+' / '+replayFrames.length;}}function stepFrame(d){{setFrame(replayIndex+d);}}function toggleReplay(){{const b=document.getElementById('play-btn');if(replayTimer){{clearInterval(replayTimer);replayTimer=null;b.textContent='Play';return;}}b.textContent='Pause';replayTimer=setInterval(()=>setFrame((replayIndex+1)%replayFrames.length),700);}}</script></section>")
    parts.append("<section><h2>Board Timeline</h2><div class='grid'>")
    for name in svg_names[-36:]:
        parts.append(f"<div class='card'><img src='{escape(name)}' alt='{escape(name)}'></div>")
    if not svg_names:
        snap = latest_state.get("snapshot", {}) if isinstance(latest_state, dict) else {}
        parts.append("<pre>" + escape(json.dumps(snap, ensure_ascii=False, indent=2, default=str)) + "</pre>")
    parts.append("</div></section>")
    parts.append("<section><h2>Decision Timeline</h2><table><tr><th>#</th><th>candidate</th><th>action</th><th>confidence</th><th>rationale</th></tr>")
    for i, d in enumerate(decisions):
        parts.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(i, escape(str(d.get("candidate_id", ""))), escape(str(d.get("action_text", ""))), escape(str(d.get("confidence", ""))), escape(str(d.get("rationale", ""))[:500])))
    parts.append("</table></section>")
    parts.append("<section><h2>Latest Candidate Ranking</h2>")
    for c in last_candidates[:12] if isinstance(last_candidates, list) else []:
        parts.append("<div class='card'><b>{}</b> {} score={}<br><span class='small'>{}</span></div>".format(escape(str(c.get("candidate_id", ""))), escape(str(c.get("action_text", ""))), escape(str(c.get("score", ""))), escape(" | ".join(map(str, c.get("reasons", []))))))
    parts.append("</section>")
    parts.append("<section><h2>Memory / Skill Events</h2><pre>" + escape(json.dumps(memory_events[-80:], ensure_ascii=False, indent=2, default=str)) + "</pre></section>")
    if eval_data:
        parts.append("<section><h2>Evaluation Artifacts</h2><div class='grid'>")
        for game, data in eval_data.items():
            parts.append(f"<div class='card'><h3>{escape(game)}</h3><pre>{escape(json.dumps(data, ensure_ascii=False, indent=2, default=str))}</pre></div>")
        parts.append("</div></section>")
    parts.append("<section><h2>Raw Event Counts</h2><pre>" + escape(json.dumps({"events": len(events), "board_frames": len(frames), "decisions": len(decisions)}, indent=2)) + "</pre></section>")
    parts.append("</main></body></html>")
    path = report_path(output_dir, "report", ".html", timestamp=timestamp, timestamped=timestamped)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render static TextArena trace HTML, SVG board frames, and Markdown trajectory.")
    parser.add_argument("--trace-dir", default="workspace/textarena_runs/latest")
    parser.add_argument("--eval-root", default="workspace/eval_runs")
    parser.add_argument("--output-dir", default="workspace/textarena_reports/latest")
    parser.add_argument("--no-timestamp", action="store_true", help="write legacy latest filenames instead of timestamped report filenames")
    args = parser.parse_args()
    stamp = artifact_timestamp()
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = read_jsonl(trace_dir / "events.jsonl")
    latest_state = read_json(trace_dir / "latest_state.json") or {}
    state_frame_rows = read_jsonl(trace_dir / "state_frames.jsonl")
    frames = []
    for row in state_frame_rows:
        snap = row.get("snapshot") if isinstance(row, dict) else None
        if isinstance(snap, dict) and snap.get("board") is not None:
            frames.append({"event": "state_frame", "created_at": row.get("updated_at"), "snapshot": snap})
    if not frames:
        frames = board_frames(events, latest_state)
    svg_names = write_svg_frames(frames, output_dir)
    report, markdown = render_html(trace_dir, output_dir, load_eval(Path(args.eval_root)), svg_names, timestamp=stamp, timestamped=not args.no_timestamp)
    payload = {"report": str(report), "markdown": str(markdown), "frames": len(svg_names), "frames_dir": str(output_dir / "frames")}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
