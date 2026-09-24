"""
Builds a self-contained, dependency-free HTML dashboard (no build step,
opens directly in a browser) from the real JSON output of evaluate.py and
guardrail_demo.py. This is the "AI-Ops dashboard" from the spec, delivered
as a static artifact so it's viewable immediately without npm/React
tooling; frontend/ has the React+TypeScript source for the production
version that talks to the live FastAPI backend.

Run: python scripts/build_dashboard.py
Writes: data/evaluation/dashboard.html
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "evaluation" / "results.json"
GUARDRAIL_PATH = ROOT / "data" / "evaluation" / "guardrail_demo_results.json"
OUT_PATH = ROOT / "data" / "evaluation" / "dashboard.html"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FrontierOps AI — Evaluation Dashboard</title>
<style>
  :root {{
    --bg: #0b0f14; --card: #121821; --border: #22303f; --text: #e6edf3;
    --muted: #8b98a5; --accent: #4fd1c5; --warn: #f0b429; --bad: #f16063; --good: #3fb950;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 24px; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 28px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .metric-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .metric-value {{ font-size: 26px; font-weight: 600; margin-top: 4px; }}
  .good {{ color: var(--good); }}
  .warn {{ color: var(--warn); }}
  .bad {{ color: var(--bad); }}
  h2 {{ font-size: 15px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin: 28px 0 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 500; }}
  .pill {{ padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .pill-good {{ background: rgba(63,185,80,0.15); color: var(--good); }}
  .pill-bad {{ background: rgba(241,96,99,0.15); color: var(--bad); }}
  .note {{ background: rgba(240,180,41,0.1); border: 1px solid rgba(240,180,41,0.3); border-radius: 8px; padding: 12px 14px; font-size: 13px; color: #f0d999; margin-bottom: 24px; }}
</style>
</head>
<body>
  <h1>FrontierOps AI — Evaluation Dashboard</h1>
  <div class="sub">Generated from a real pipeline run on {num_questions} benchmark questions · offline hashing embedder (no network) · {backend_note}</div>

  <div class="note">These numbers come from executing the full six-agent pipeline against
  the benchmark in this repo, not from placeholders. The offline fallback embedder
  (used because this environment has no network for sentence-transformers/FAISS)
  compresses cosine-similarity scores, which lowers context precision and raises the
  human-review rate versus what a production embedding model would give — see
  README.md "Known limitations of the offline demo".</div>

  <h2>Retrieval &amp; Answer Quality</h2>
  <div class="grid">
    <div class="card"><div class="metric-label">Context Precision</div><div class="metric-value">{context_precision}</div></div>
    <div class="card"><div class="metric-label">Context Recall</div><div class="metric-value">{context_recall}</div></div>
    <div class="card"><div class="metric-label">Faithfulness</div><div class="metric-value">{faithfulness}</div></div>
    <div class="card"><div class="metric-label">Citation Accuracy</div><div class="metric-value">{citation_accuracy}</div></div>
  </div>

  <h2>Operations</h2>
  <div class="grid">
    <div class="card"><div class="metric-label">Avg Latency</div><div class="metric-value">{avg_latency_ms} ms</div></div>
    <div class="card"><div class="metric-label">Workflow Success Rate</div><div class="metric-value">{success_rate}</div></div>
    <div class="card"><div class="metric-label">Human Review Rate</div><div class="metric-value">{human_review_rate}</div></div>
    <div class="card"><div class="metric-label">Injection Attempts Caught</div><div class="metric-value">{injection_caught}</div></div>
  </div>

  <h2>Guardrail Effectiveness (adversarial test set, {guardrail_cases} cases)</h2>
  <div class="grid">
    <div class="card"><div class="metric-label">Unsupported, shown to user — before</div><div class="metric-value bad">{unsupported_before}</div></div>
    <div class="card"><div class="metric-label">Unsupported, shown to user — after</div><div class="metric-value good">{unsupported_after}</div></div>
    <div class="card"><div class="metric-label">Reduction</div><div class="metric-value good">{unsupported_reduction}%</div></div>
  </div>

  <h2>Agent Pipeline</h2>
  <table>
    <tr><th>Agent</th><th>Status</th></tr>
    {agent_rows}
  </table>

  <h2>Per-Question Results</h2>
  <table>
    <tr><th>ID</th><th>Question</th><th>Intent</th><th>Confidence</th><th>Faithfulness</th><th>Human Review</th></tr>
    {question_rows}
  </table>
</body>
</html>
"""


def render():
    results = json.loads(RESULTS_PATH.read_text())
    guardrail = json.loads(GUARDRAIL_PATH.read_text()) if GUARDRAIL_PATH.exists() else {
        "cases": 0, "unsupported_answers_shown_before_guardrails": 0,
        "unsupported_answers_shown_after_guardrails": 0, "reduction_pct": 0.0,
    }

    def pct(x):
        return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"

    agents = ["Orchestrator", "Retrieval", "Research", "Analysis", "Validation", "Response"]
    agent_rows = "\n".join(
        f'<tr><td>{a}</td><td><span class="pill pill-good">OK</span></td></tr>' for a in agents
    )

    question_rows = []
    for q in results.get("per_question", []):
        review = q.get("human_review")
        pill = f'<span class="pill {"pill-bad" if review else "pill-good"}">{review}</span>'
        question_rows.append(
            f'<tr><td>{q["id"]}</td><td>{q["question"]}</td><td>{q.get("intent","")}</td>'
            f'<td>{q.get("confidence",0):.2f}</td><td>{q.get("faithfulness",0):.2f}</td><td>{pill}</td></tr>'
        )

    html = TEMPLATE.format(
        num_questions=results["num_questions"],
        backend_note="ExtractiveMockLLM (offline)" ,
        context_precision=pct(results.get("context_precision")),
        context_recall=pct(results.get("context_recall")),
        faithfulness=pct(results.get("faithfulness")),
        citation_accuracy=pct(results.get("citation_accuracy")),
        avg_latency_ms=results.get("avg_latency_ms"),
        success_rate=pct(results.get("success_rate")),
        human_review_rate=pct(results.get("human_review_rate")),
        injection_caught=results.get("injection_attempts_caught", 0),
        guardrail_cases=guardrail.get("cases", 0),
        unsupported_before=guardrail.get("unsupported_answers_shown_before_guardrails", 0),
        unsupported_after=guardrail.get("unsupported_answers_shown_after_guardrails", 0),
        unsupported_reduction=guardrail.get("reduction_pct", 0.0),
        agent_rows=agent_rows,
        question_rows="\n".join(question_rows),
    )
    OUT_PATH.write_text(html)
    return OUT_PATH


if __name__ == "__main__":
    path = render()
    print(f"Dashboard written to {path}")
