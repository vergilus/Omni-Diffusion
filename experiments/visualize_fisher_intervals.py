"""Visualize Fisher-Rao interval summaries as a self-contained interactive HTML.

Reads one or more ``fisher_summary*.json`` files produced by
``experiments/check_fisher.py`` and writes a single HTML file. Task type
(ASR / TTS / VQA) is the top-level tab. Each task draws one sequence-level
bar per interval (``num_intervals = num_time_steps - 1``).
The output uses no third-party Python or JavaScript libraries, so it can be
generated on a headless server and opened in any browser.
"""

import argparse
import json
from pathlib import Path


METRICS = {
    "sequence": {
        "label": "Sequence Interval",
        "mean": "sequence_interval_mean",
        "std": "sequence_interval_std",
        "negative": "sequence_interval_negative_fraction",
        "description": "len*(c*(theta_t-theta_0))^2 - sum_j dF(prob_0, prob_t)_j^2",
    },
}

TASK_COLORS = {"ASR": "#4C78A8", "TTS": "#F58518", "VQA": "#54A24B"}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fisher-Rao Interval Visualization</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; background: #fafafa; color: #222; }
  header { padding: 16px 24px 8px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .meta { font-size: 13px; color: #666; }
  .controls { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; padding: 8px 24px; }
  .tabs { display: inline-flex; border: 1px solid #ccc; border-radius: 6px; overflow: hidden; }
  .tabs button { border: none; background: #eee; padding: 8px 18px; cursor: pointer; font-size: 14px; }
  .tabs button.active { background: #4C78A8; color: #fff; }
  .tabs.tasks button.active { background: #2f4f75; }
  .control-group { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; }
  select, label { font-size: 14px; }
  #chart { padding: 0 24px 24px; position: relative; }
  #tooltip { position: absolute; pointer-events: none; background: rgba(30,30,30,.92); color: #fff;
             font-size: 12px; padding: 8px 10px; border-radius: 4px; display: none; max-width: 320px; z-index: 10; }
  svg text { font-size: 11px; fill: #444; }
  .axis-label { font-size: 13px; }
  .range-label { font-size: 11px; fill: #888; font-style: italic; }
  .desc { padding: 0 24px 8px; font-size: 12px; color: #888; font-style: italic; }
</style>
</head>
<body>
<header>
  <h1>Fisher-Rao Interval Visualization</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="controls">
  <div class="tabs tasks" id="task-tabs"></div>
  <div class="control-group">
    <label for="file-select">Summary:</label>
    <select id="file-select"></select>
  </div>
  <div class="control-group">
    <label><input type="checkbox" id="log-scale"> log y</label>
    <label><input type="checkbox" id="show-std" checked> show ±std</label>
  </div>
</div>
<div class="desc" id="desc"></div>
<div id="chart"><div id="tooltip"></div><svg id="svg" width="100%" height="560"></svg></div>
<script id="fisher-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("fisher-data").textContent);
const METRICS = __METRICS__;
const state = { task: null, fileIndex: 0, logScale: false, showStd: true };

  const svg = document.getElementById("svg");
  const tooltip = document.getElementById("tooltip");
  const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs) {
  const node = document.createElementNS(NS, tag);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  return node;
}

function fmt(v, digits) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-3) return v.toExponential(2);
  return v.toFixed(digits === undefined ? 4 : digits);
}

function currentFile() { return DATA[state.fileIndex]; }

function currentTask() {
  const task = currentFile().tasks.find(t => t.name === state.task);
  return task || null;
}

function buildControls() {
  const file = currentFile();
  if (!file.tasks.some(t => t.name === state.task)) state.task = file.tasks[0].name;

  const taskTabs = document.getElementById("task-tabs");
  taskTabs.innerHTML = "";
  for (const t of file.tasks) {
    const btn = document.createElement("button");
    btn.textContent = t.name;
    btn.className = t.name === state.task ? "active" : "";
    btn.onclick = () => { state.task = t.name; buildControls(); render(); };
    taskTabs.appendChild(btn);
  }

  const sel = document.getElementById("file-select");
  sel.innerHTML = "";
  DATA.forEach((f, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = f.name;
    sel.appendChild(opt);
  });
  sel.value = state.fileIndex;
  sel.onchange = () => { state.fileIndex = +sel.value; buildControls(); render(); };

  document.getElementById("meta").textContent =
    `samples=${file.num_samples}  time_steps=${file.num_time_steps}  ` +
    `intervals=${file.num_intervals}  time_grid=${file.time_grid}  c=${file.interval_c}` +
    `  samples(task)=${currentTask() ? currentTask().num_samples : "-"}`;
  document.getElementById("desc").textContent = METRICS.sequence.description;
}

function niceTicks(min, max, count) {
  if (min === max) return [min];
  const span = max - min;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-12 * Math.max(1, Math.abs(max)); v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return ticks;
}

function render() {
  const metric = METRICS.sequence;
  const width = svg.clientWidth || 1200;
  const height = 560;
  const margin = { top: 30, right: 24, bottom: 48, left: 110 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const defs = el("defs", {});
  const hatch = el("pattern", {
    id: "negative-hatch", width: 8, height: 8,
    patternUnits: "userSpaceOnUse", patternTransform: "rotate(0)",
  });
  hatch.appendChild(el("path", {
    d: "M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6",
    stroke: "#ffffff", "stroke-width": 1.2, fill: "none",
  }));
  defs.appendChild(hatch);
  svg.appendChild(defs);

  const file = currentFile();
  const task = currentTask();
  const n = file.num_intervals;
  if (!task) {
    const text = el("text", { x: width / 2, y: height / 2, "text-anchor": "middle" });
    text.textContent = "No task selected";
    svg.appendChild(text);
    return;
  }

  const means = task.intervals.map(iv => iv[metric.mean]);
  const stds = task.intervals.map(iv => iv[metric.std]);
  const lows = means.map((m, i) => m - (state.showStd ? stds[i] : 0));
  const highs = means.map((m, i) => m + (state.showStd ? stds[i] : 0));
  const dataMin = Math.min(...lows), dataMax = Math.max(...highs);
  const span = dataMax - dataMin || Math.abs(dataMax) || 1;
  let yMin = dataMin - span * 0.05, yMax = dataMax + span * 0.05;
  yMin = Math.min(yMin, 0);
  yMax = Math.max(yMax, 0);

  // symlog transform: logarithmic away from zero, linear within ±linthresh,
  // so both signs and magnitudes spanning several decades stay visible.
  let linthresh = 0;
  if (state.logScale) {
    const pos = [];
    for (const arr of [means, lows, highs]) for (const v of arr) if (v > 0) pos.push(v);
    linthresh = pos.length ? Math.min(...pos) : Math.max(1e-300, span / 1e6);
  }
  const T = v => state.logScale
    ? Math.sign(v) * linthresh * Math.log10(1 + Math.abs(v) / linthresh)
    : v;
  const tMin = T(yMin), tMax = T(yMax);

  const xOf = i => margin.left + (i / n) * plotW;
  const yOf = v => margin.top + plotH * (1 - (T(v) - tMin) / (tMax - tMin));

  let yTicks;
  if (state.logScale) {
    yTicks = [0, linthresh, -linthresh];
    for (let k = 2; ; k++) {
      const v = linthresh * (Math.pow(10, k) - 1);
      if (v > yMax && v > -yMin) break;
      if (v <= yMax) yTicks.push(v);
      if (v <= -yMin) yTicks.push(-v);
    }
  } else {
    yTicks = niceTicks(yMin, yMax, 8);
  }
  for (const t of yTicks) {
    if (t < yMin - 1e-18 || t > yMax + 1e-18) continue;
    const y = yOf(t);
    svg.appendChild(el("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: "#e0e0e0" }));
    const label = el("text", { x: margin.left - 6, y: y + 4, "text-anchor": "end" });
    label.textContent = fmt(t);
    svg.appendChild(label);
  }
  // Emphasized zero line: bars start here, negative values extend downward.
  const yZero = yOf(0);
  svg.appendChild(el("line", { x1: margin.left, x2: width - margin.right, y1: yZero, y2: yZero, stroke: "#333", "stroke-width": 1.5 }));
  for (let i = 0; i <= 10; i++) {
    const idx = Math.round((i / 10) * (n - 1));
    const x = xOf(idx + 0.5);
    svg.appendChild(el("line", { x1: x, x2: x, y1: margin.top + plotH, y2: margin.top + plotH + 5, stroke: "#888" }));
    const label = el("text", { x: x, y: margin.top + plotH + 20, "text-anchor": "middle" });
    label.textContent = idx + 1;
    svg.appendChild(label);
  }
  svg.appendChild(el("line", { x1: margin.left, x2: width - margin.right, y1: margin.top + plotH, y2: margin.top + plotH, stroke: "#888" }));

  // Explicit numeric range of the y axis, shown next to the axis title.
  const rangeText = el("text", { x: margin.left - 6, y: margin.top - 16, "text-anchor": "end", class: "range-label" });
  rangeText.textContent = "y range: [" + fmt(yMin) + ", " + fmt(yMax) + "]" + (state.logScale ? " (log)" : "");
  svg.appendChild(rangeText);

  const yLabel = el("text", { x: 16, y: margin.top + plotH / 2, "text-anchor": "middle", class: "axis-label", transform: `rotate(-90 16 ${margin.top + plotH / 2})` });
  yLabel.textContent = metric.label + (state.logScale ? " (log scale)" : "");
  svg.appendChild(yLabel);
  const xLabel = el("text", { x: margin.left + plotW / 2, y: height - 8, "text-anchor": "middle", class: "axis-label" });
  xLabel.textContent = "interval index (step = interval_index + 1)";
  svg.appendChild(xLabel);

  const slot = plotW / n;
  const barW = Math.max(0.5, slot - Math.max(0.5, slot * 0.15));
  for (let i = 0; i < n; i++) {
    const iv = task.intervals[i];
    const mean = iv[metric.mean];
    const std = state.showStd ? iv[metric.std] : 0;
    // Fade intervals with a larger negative-sample fraction while retaining
    // the task hue. A fully negative interval remains visible at low opacity.
    const negative = Math.max(0, Math.min(1, Number(iv[metric.negative]) || 0));
    const opacity = 0.18 + 0.72 * (1 - negative);
    const x = xOf(i) + (slot - barW) / 2;
    const yTop = yOf(mean + std);
    const yBase = yOf(state.logScale ? Math.max(mean, yMin) : yMin);
    const rect = el("rect", {
      x: x, y: yTop, width: barW, height: Math.max(0.5, yBase - yTop),
      fill: task.color, opacity: opacity,
    });
    rect.addEventListener("mousemove", ev => showTooltip(ev, i));
    rect.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(rect);
    const texture = el("rect", {
      x: x, y: yTop, width: barW, height: Math.max(0.5, yBase - yTop),
      fill: "url(#negative-hatch)", opacity: 0.78 * negative,
      "pointer-events": "none",
    });
    svg.appendChild(texture);
  }

  function showTooltip(ev, i) {
    const iv = task.intervals[i];
    tooltip.innerHTML =
      `<div><b>${task.name}</b> &nbsp; interval ${i + 1} / step ${iv.step || i + 1}</div>` +
      `<div>mean=${fmt(iv[metric.mean])} &plusmn; ${fmt(iv[metric.std])} ` +
      `(neg ${(iv[metric.negative] * 100).toFixed(2)}%)</div>` +
      `<div>&alpha;: ${iv.alpha_start.toFixed(4)} &rarr; ${iv.alpha_end.toFixed(4)}` +
      ` &nbsp; &Delta;&theta;: ${iv.delta_theta.toExponential(2)}</div>` +
      `<div>target positions: ${iv.target_position_count}</div>`;
    tooltip.style.display = "block";
    const host = document.getElementById("chart").getBoundingClientRect();
    let px = ev.clientX - host.left + 14, py = ev.clientY - host.top - 10;
    if (px + 330 > host.width) px -= 350;
    tooltip.style.left = px + "px";
    tooltip.style.top = py + "px";
  }
  function hideTooltip() { tooltip.style.display = "none"; }
}

function init() {
  state.task = DATA[0].tasks[0].name;
  document.getElementById("log-scale").onchange = e => { state.logScale = e.target.checked; render(); };
  document.getElementById("show-std").onchange = e => { state.showStd = e.target.checked; render(); };
  buildControls();
  render();
  window.addEventListener("resize", render);
}
init();
</script>
</body>
</html>
"""


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    intervals = summary.get("intervals")
    if intervals is not None:
        by_task = {"all": {"intervals": intervals}}
    else:
        by_task = summary.get("by_task", {})
    if not by_task:
        raise ValueError(f"{path} contains no interval data")
    tasks = []
    for task_name in sorted(by_task):
        stats = by_task[task_name]
        task_intervals = stats.get("intervals")
        if not task_intervals:
            continue
        tasks.append(
            {
                "name": task_name,
                "color": TASK_COLORS.get(task_name, "#999999"),
                "num_samples": int(stats.get("num_samples", 0)),
                "intervals": task_intervals,
            }
        )
    if not tasks:
        raise ValueError(f"{path} contains no interval data")
    num_intervals = len(tasks[0]["intervals"])
    for task in tasks:
        if len(task["intervals"]) != num_intervals:
            raise ValueError(f"{path} task {task['name']} interval length mismatch")
    return {
        "name": path.name,
        "num_samples": int(summary.get("num_samples", 0)),
        "num_time_steps": int(summary.get("num_time_steps", num_intervals + 1)),
        "num_intervals": num_intervals,
        "time_grid": summary.get("time_grid", "unknown"),
        "interval_c": summary.get("interval_c", 1.0),
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="fisher_summary*.json file(s) produced by experiments/check_fisher.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("fisher_intervals.html"),
        help="output HTML path (default: fisher_intervals.html)",
    )
    args = parser.parse_args()

    files = [load_summary(path) for path in args.summaries]

    metrics_json = json.dumps(
        {
            key: {
                "label": value["label"],
                "mean": value["mean"],
                "std": value["std"],
                "negative": value["negative"],
                "description": value["description"],
            }
            for key, value in METRICS.items()
        }
    )
    html = (
        HTML_TEMPLATE.replace("__DATA__", json.dumps(files))
        .replace("__METRICS__", metrics_json)
    )
    args.output.write_text(html, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    # python experiments/visualize_fisher_intervals.py test_out/fisher_analysis/fisher_summary.json -o  test_out/fisher_interval.html
    main()
