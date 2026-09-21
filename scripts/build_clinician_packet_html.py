"""
Build the self-contained, blinded clinician-facing HTML review packet.

Reads docs/clinician_review/case_manifest.json for case IDs and image paths
ONLY (never the pipeline's classification fields -- this script must stay
blind to those, same as the clinician). Embeds each case image as a base64
data URI so the output is a single, shareable, offline-viewable HTML file.

Usage:
    python scripts/build_clinician_packet_html.py [output_path]
"""

import base64
import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REVIEW_DIR = os.path.join(BASE_DIR, "docs", "clinician_review")
MANIFEST_PATH = os.path.join(REVIEW_DIR, "case_manifest.json")
DEFAULT_OUT = os.path.join(REVIEW_DIR, "clinician_packet.html")


def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


CASE_CARD_TEMPLATE = """
    <section class="case-card" id="case-{case_id}">
      <div class="case-head">
        <span class="case-id">Case {case_id}</span>
      </div>
      <div class="case-body">
        <img class="case-trace" src="data:image/png;base64,{img_b64}" alt="CTG trace, {case_id}">
        <div class="case-form">
          <div class="field">
            <label for="{case_id}-baseline">Estimated baseline (bpm)</label>
            <input type="number" id="{case_id}-baseline" name="{case_id}-baseline" min="50" max="240">
          </div>
          <div class="field">
            <label for="{case_id}-var">Variability impression</label>
            <select id="{case_id}-var" name="{case_id}-var">
              <option value=""></option>
              <option>Reduced (&lt;5 bpm)</option>
              <option>Normal (5&ndash;25 bpm)</option>
              <option>Increased / saltatory (&gt;25 bpm)</option>
            </select>
          </div>
          <div class="field">
            <label for="{case_id}-accel">Accelerations (count)</label>
            <input type="number" id="{case_id}-accel" name="{case_id}-accel" min="0">
          </div>
          <div class="field decel-field">
            <span class="field-label">Decelerations observed (count by type)</span>
            <div class="decel-grid">
              <label>Early <input type="number" min="0" name="{case_id}-decel-early"></label>
              <label>Late <input type="number" min="0" name="{case_id}-decel-late"></label>
              <label>Variable <input type="number" min="0" name="{case_id}-decel-variable"></label>
              <label>Prolonged <input type="number" min="0" name="{case_id}-decel-prolonged"></label>
            </div>
          </div>
          <div class="field">
            <span class="field-label">Your FIGO category</span>
            <div class="radio-row">
              <label><input type="radio" name="{case_id}-figo" value="normal"> Normal</label>
              <label><input type="radio" name="{case_id}-figo" value="suspicious"> Suspicious</label>
              <label><input type="radio" name="{case_id}-figo" value="pathological"> Pathological</label>
            </div>
          </div>
          <div class="field">
            <span class="field-label">Confidence in this read</span>
            <div class="radio-row">
              <label><input type="radio" name="{case_id}-confidence" value="low"> Low</label>
              <label><input type="radio" name="{case_id}-confidence" value="medium"> Medium</label>
              <label><input type="radio" name="{case_id}-confidence" value="high"> High</label>
            </div>
          </div>
          <div class="field field-wide">
            <label for="{case_id}-notes">Notes (anything that stood out, including if the trace looks artifactual)</label>
            <textarea id="{case_id}-notes" name="{case_id}-notes" rows="2"></textarea>
          </div>
        </div>
      </div>
    </section>
"""


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    case_cards = []
    for case in manifest:
        case_id = case["case_id"]
        img_path = os.path.join(REVIEW_DIR, case["image"])
        img_b64 = b64_image(img_path)
        case_cards.append(CASE_CARD_TEMPLATE.format(case_id=case_id, img_b64=img_b64))

    html = HTML_TEMPLATE.format(
        n_cases=len(manifest),
        case_cards="\n".join(case_cards),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Done] Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB, {len(manifest)} cases)")


HTML_TEMPLATE = """<title>CTG Trace Review &mdash; Clinician Packet</title>
<style>
  :root {{
    --bg: #EEF2F3;
    --surface: #FFFFFF;
    --surface-alt: #E4EAEC;
    --border: #D3DBDE;
    --text: #1B2427;
    --text-muted: #57676C;
    --accent: #2A6FA0;
    --accent-strong: #184A6E;
    --accent-soft: #DCEAF3;
    --input-bg: #FFFFFF;
    --input-border: #B9C4C8;
    --shadow: 0 1px 2px rgba(20,30,32,0.06), 0 6px 18px rgba(20,30,32,0.05);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #0F1417; --surface: #171F23; --surface-alt: #1D262A; --border: #2B363A;
      --text: #E7ECEE; --text-muted: #93A4A9; --accent: #6FB6E0; --accent-strong: #9ED0EC;
      --accent-soft: #17323F; --input-bg: #1B2226; --input-border: #37454A;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 6px 20px rgba(0,0,0,0.35);
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #0F1417; --surface: #171F23; --surface-alt: #1D262A; --border: #2B363A;
    --text: #E7ECEE; --text-muted: #93A4A9; --accent: #6FB6E0; --accent-strong: #9ED0EC;
    --accent-soft: #17323F; --input-bg: #1B2226; --input-border: #37454A;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 6px 20px rgba(0,0,0,0.35);
  }}
  :root[data-theme="light"] {{
    --bg: #EEF2F3; --surface: #FFFFFF; --surface-alt: #E4EAEC; --border: #D3DBDE;
    --text: #1B2427; --text-muted: #57676C; --accent: #2A6FA0; --accent-strong: #184A6E;
    --accent-soft: #DCEAF3; --input-bg: #FFFFFF; --input-border: #B9C4C8;
    --shadow: 0 1px 2px rgba(20,30,32,0.06), 0 6px 18px rgba(20,30,32,0.05);
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
    font-size: 16px; line-height: 1.6;
  }}
  h1, h2, .case-id, .field-label, label {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; padding: 40px 28px 100px; }}

  header.intro {{ margin-bottom: 40px; }}
  .eyebrow {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-weight: 800; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent-strong); margin-bottom: 10px;
  }}
  h1 {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-weight: 800; font-size: 1.9rem; letter-spacing: -0.01em; margin: 0 0 14px;
    text-wrap: balance;
  }}
  .lede {{ font-size: 1.05rem; max-width: 68ch; color: var(--text); }}
  p {{ max-width: 68ch; margin: 0 0 14px; }}

  .callout {{
    background: var(--accent-soft); border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--border));
    border-radius: 6px; padding: 16px 18px; margin: 22px 0; max-width: 68ch;
  }}
  .callout .callout-label {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-weight: 800; font-size: 0.68rem; letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--accent-strong); display: block; margin-bottom: 8px;
  }}
  .callout p:last-child {{ margin-bottom: 0; }}

  ol.steps {{ max-width: 68ch; padding-left: 22px; }}
  ol.steps li {{ margin-bottom: 10px; }}

  h2 {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-weight: 800; font-size: 1.05rem; letter-spacing: 0.02em; text-transform: uppercase;
    color: var(--accent-strong); margin: 46px 0 16px; padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }}

  .case-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    box-shadow: var(--shadow); margin-bottom: 28px; overflow: hidden;
    break-inside: avoid; page-break-inside: avoid;
  }}
  .case-head {{
    background: var(--surface-alt); padding: 10px 18px; border-bottom: 1px solid var(--border);
  }}
  .case-id {{
    font-weight: 800; letter-spacing: 0.04em; font-variant-numeric: tabular-nums;
    color: var(--accent-strong); font-size: 0.95rem;
  }}
  .case-body {{ padding: 18px; }}
  .case-trace {{
    width: 100%; height: auto; border: 1px solid var(--border); border-radius: 4px;
    display: block; margin-bottom: 18px; background: #fff;
  }}

  .case-form {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px 20px;
  }}
  .field {{ display: flex; flex-direction: column; gap: 6px; }}
  .field-wide {{ grid-column: 1 / -1; }}
  .field label, .field-label {{
    font-size: 0.78rem; font-weight: 600; color: var(--text-muted);
  }}
  input[type="number"], select, textarea {{
    font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
    font-size: 0.92rem; padding: 7px 9px; border-radius: 5px;
    border: 1px solid var(--input-border); background: var(--input-bg); color: var(--text);
  }}
  select {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  textarea {{ font-family: Georgia, serif; resize: vertical; }}
  input:focus, select:focus, textarea:focus {{
    outline: 2px solid var(--accent); outline-offset: 1px;
  }}

  .decel-field {{ grid-column: 1 / -1; }}
  .decel-grid {{ display: flex; flex-wrap: wrap; gap: 14px; }}
  .decel-grid label {{
    display: flex; align-items: center; gap: 6px; font-size: 0.82rem; color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  .decel-grid input {{ width: 56px; }}

  .radio-row {{ display: flex; flex-wrap: wrap; gap: 14px; }}
  .radio-row label {{
    display: flex; align-items: center; gap: 6px; font-size: 0.86rem; font-weight: 400;
    color: var(--text); font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}

  .summary-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    box-shadow: var(--shadow); padding: 20px 22px; margin: 24px 0 60px;
  }}
  .summary-card .field {{ margin-bottom: 18px; max-width: 68ch; }}
  .summary-card label {{ font-size: 0.9rem; font-weight: 700; color: var(--text); margin-bottom: 8px; }}
  .summary-card textarea {{ width: 100%; max-width: 68ch; }}

  footer {{ color: var(--text-muted); font-size: 0.82rem; max-width: 68ch; margin-top: 40px; }}

  @media print {{
    :root {{ --bg: #fff; --surface: #fff; --surface-alt: #f3f3f3; --text: #000; --text-muted: #333; --border: #999; }}
    body {{ background: #fff; }}
    .case-card {{ box-shadow: none; page-break-after: always; }}
  }}

  ::selection {{ background: var(--accent-soft); }}
</style>

<div class="wrap">
  <header class="intro">
    <div class="eyebrow">CTG Trace Review &mdash; Clinician Packet</div>
    <h1>Does this look right to you?</h1>
    <p class="lede">This project uses a piece of software that reads fetal heart rate (FHR) and uterine
    contraction (UC) traces and automatically estimates the same things an obstetrician looks for &mdash;
    baseline rate, variability, accelerations, decelerations &mdash; and from those, a FIGO 2015 category
    (Normal / Suspicious / Pathological). We want to know whether that automated reading actually matches
    what a clinician would see. That's the only thing we're asking you to judge here.</p>

    <div class="callout">
      <span class="callout-label">Why this matters</span>
      <p>Think of the software as a resident being graded not just on a final yes/no call, but on whether
      its own stated vital signs (baseline, variability, decelerations) are internally consistent and match
      what an expert actually sees on the same trace. Your read is the only way we can check that &mdash;
      the software's own numbers can't grade themselves.</p>
    </div>

    <h2 style="margin-top:32px;">How to use this packet</h2>
    <ol class="steps">
      <li><strong>Read each case cold.</strong> No clinical history, no prior classification is shown &mdash;
      just the trace, the way you'd see it on a monitor. That's intentional: it keeps your read independent.</li>
      <li>For each case, fill in your <strong>estimated baseline, variability impression, accelerations,
      decelerations (by type if you can tell), your own FIGO category, and how confident you are.</strong></li>
      <li>Please go in order and don't skip around or revisit earlier answers after seeing later cases &mdash;
      that keeps each read independent of the others.</li>
      <li>There are {n_cases} cases. If you need to split this across more than one sitting, that's fine &mdash;
      just try not to re-review a case you've already scored.</li>
      <li>A couple of summary questions are at the very end, after the last case.</li>
    </ol>

    <div class="callout">
      <span class="callout-label">Note on filling this in</span>
      <p>This page doesn't submit or save anywhere &mdash; it's a local, fillable document. Please save/export
      it (e.g. print to PDF, or take a copy of the completed page) and send it back the way you'd normally
      share files with us. Typed entries will stay in the fields as long as you don't close the tab, but
      won't persist beyond that on their own.</p>
    </div>
  </header>

  <h2>Cases</h2>
{case_cards}

  <h2>A couple of last questions</h2>
  <div class="summary-card">
    <div class="field">
      <label>Overall, which type of error would concern you more in practice: missing a real case of
      fetal distress, or a false alarm that leads to unnecessary intervention? There's no wrong answer &mdash;
      we're asking because it directly affects how strict we should set the system's alert threshold.</label>
      <textarea rows="3"></textarea>
    </div>
    <div class="field">
      <label>Anything about the trace format itself (scale, grid, what was shown or not shown) that made
      this harder to read than a real monitor or paper strip would be?</label>
      <textarea rows="3"></textarea>
    </div>
  </div>

  <footer>
    Thank you for taking the time to do this &mdash; it's the piece of validation that nothing in the
    software itself can provide. If anything here was confusing or you'd rather discuss a case verbally
    instead of writing it up, that's completely fine too.
  </footer>
</div>
"""


if __name__ == "__main__":
    main()
