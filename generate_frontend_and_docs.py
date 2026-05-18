from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from notebook_bridge import DATA_PATH, NOTEBOOK_PATH, load_notebook_namespace


BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "index.html"
PDF_PATH = BASE_DIR / "HR_Attrition_Code_Documentation.pdf"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def table_html(df: pd.DataFrame, limit: int | None = None) -> str:
    data = df.head(limit) if limit else df
    return data.to_html(index=False, border=0, classes="data-table", justify="left", escape=True)


def build_html(ns: dict) -> str:
    raw_df = pd.read_csv(DATA_PATH)
    results_df = ns.get("results_df", pd.DataFrame())
    smote_results_df = ns.get("smote_results_df", pd.DataFrame())
    target_counts = raw_df["Attrition"].value_counts().rename_axis("Attrition").reset_index(name="Count")
    target_counts["Percentage"] = target_counts["Count"].div(len(raw_df)).map(fmt_pct)
    null_counts = raw_df.isnull().sum()
    missing_df = (
        null_counts[null_counts > 0]
        .rename_axis("Column")
        .reset_index(name="Missing Values")
        .sort_values("Missing Values", ascending=False)
    )
    if missing_df.empty:
        missing_df = pd.DataFrame([{"Column": "None", "Missing Values": 0}])

    numeric_summary = raw_df.describe().T.reset_index().rename(columns={"index": "Column"})
    numeric_summary = numeric_summary[["Column", "count", "mean", "std", "min", "50%", "max"]].round(2)
    categorical_cols = raw_df.select_dtypes(include="object").columns.tolist()
    numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
    dropped_columns = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

    importance = pd.Series(ns["rf"].feature_importances_, index=ns["X"].columns).sort_values(ascending=False)
    importance_df = importance.head(10).reset_index()
    importance_df.columns = ["Feature", "Importance"]
    importance_df["Importance"] = importance_df["Importance"].round(4)
    label_encoders = ns["label_encoders"]
    scaler = ns["scaler"]
    lr = ns["lr"]
    feature_payload = []
    for idx, feature in enumerate(ns["X"].columns):
        if feature in label_encoders:
            classes = [str(value) for value in label_encoders[feature].classes_]
            default_value = str(raw_df[feature].mode(dropna=True).iloc[0])
            feature_payload.append(
                {
                    "name": feature,
                    "type": "categorical",
                    "classes": classes,
                    "default": default_value if default_value in classes else classes[0],
                }
            )
        else:
            series = pd.to_numeric(raw_df[feature], errors="coerce")
            feature_payload.append(
                {
                    "name": feature,
                    "type": "numeric",
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "default": float(series.median()),
                    "step": 1 if (series.dropna() % 1 == 0).all() else 0.1,
                }
            )

    model_payload = {
        "features": feature_payload,
        "scalerMean": [float(value) for value in scaler.mean_],
        "scalerScale": [float(value) for value in scaler.scale_],
        "coef": [float(value) for value in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "targetClasses": [str(value) for value in label_encoders["Attrition"].classes_],
        "modelName": "Logistic Regression",
    }
    model_json = json.dumps(model_payload)
    age_bins = pd.cut(
        raw_df["Age"],
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65"],
    )
    age_attrition = pd.crosstab(age_bins, raw_df["Attrition"]).reset_index()
    age_attrition.columns = ["Age Group"] + [str(col) for col in age_attrition.columns[1:]]
    age_attrition["Age Group"] = age_attrition["Age Group"].astype(str)
    overtime_attrition = pd.crosstab(raw_df["OverTime"], raw_df["Attrition"]).reset_index()
    overtime_attrition.columns = ["OverTime"] + [str(col) for col in overtime_attrition.columns[1:]]
    department_counts = raw_df["Department"].value_counts().reset_index()
    department_counts.columns = ["Department", "Count"]
    job_role_counts = raw_df["JobRole"].value_counts().head(8).reset_index()
    job_role_counts.columns = ["Job Role", "Count"]
    income_by_attrition = (
        raw_df.groupby("Attrition", dropna=False)["MonthlyIncome"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"MonthlyIncome": "Average Monthly Income"})
    )
    chart_payload = {
        "target": target_counts.to_dict(orient="records"),
        "department": department_counts.to_dict(orient="records"),
        "jobRole": job_role_counts.to_dict(orient="records"),
        "overtimeAttrition": overtime_attrition.to_dict(orient="records"),
        "ageAttrition": age_attrition.to_dict(orient="records"),
        "incomeByAttrition": income_by_attrition.to_dict(orient="records"),
        "missing": missing_df.to_dict(orient="records"),
        "importance": importance_df.to_dict(orient="records"),
        "beforeMetrics": results_df.round(4).to_dict(orient="records"),
        "afterMetrics": smote_results_df.round(4).to_dict(orient="records"),
    }
    chart_json = json.dumps(chart_payload)
    feature_pills = "".join(f"<li>{html.escape(col)}</li>" for col in ns["X"].columns)
    dropped_pills = "".join(f"<li>{html.escape(col)}</li>" for col in dropped_columns)
    tables = {
        "target": table_html(target_counts),
        "missing": table_html(missing_df),
        "numeric": table_html(numeric_summary, 18),
        "before": table_html(results_df.round(4)) if not results_df.empty else "<p>No results table generated.</p>",
        "after": table_html(smote_results_df.round(4)) if not smote_results_df.empty else "<p>No SMOTE table generated.</p>",
        "importance": table_html(importance_df),
    }

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HR Attrition Dashboard</title>
  <style>
    :root {
      --ink: #1d252c;
      --muted: #61717d;
      --line: #d7e0e6;
      --paper: #f4f7f7;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #b45309;
      --soft: #e8f3f0;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
      line-height: 1.55;
    }
    header {
      background: linear-gradient(135deg, #113238, #0f766e);
      color: white;
      padding: 38px 7vw 30px;
    }
    header p { max-width: 900px; color: #d9efeb; margin: 8px 0 0; }
    nav {
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(244, 247, 247, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }
    nav .inner {
      width: min(1220px, 94vw);
      margin: 0 auto;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 10px 0;
    }
    nav a {
      color: var(--accent-dark);
      text-decoration: none;
      border: 1px solid var(--line);
      background: white;
      border-radius: 999px;
      padding: 7px 12px;
      white-space: nowrap;
      font-size: 0.92rem;
    }
    main { width: min(1220px, 94vw); margin: 0 auto; padding: 24px 0 54px; }
    section { padding: 24px 0; border-bottom: 1px solid var(--line); }
    h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.35rem); letter-spacing: 0; }
    h2 { margin: 0 0 14px; font-size: 1.45rem; }
    h3 { margin: 0 0 10px; font-size: 1.02rem; color: var(--accent-dark); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
    .two { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
    .stat, .panel, .chart-card, .predictor {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(28, 48, 54, 0.06);
    }
    .stat { padding: 16px; }
    .stat b { display: block; font-size: 1.55rem; color: var(--accent); }
    .panel, .chart-card, .predictor { padding: 18px; }
    .note { color: var(--muted); }
    .chart-card canvas { width: 100%; height: 320px; display: block; }
    .mini-controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 14px; }
    .mini-controls button, button {
      border: 0;
      border-radius: 6px;
      padding: 10px 13px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    .mini-controls button.secondary { background: #e8eeee; color: var(--ink); }
    .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
    label { display: grid; gap: 5px; font-size: 0.9rem; font-weight: 700; }
    input, select {
      width: 100%;
      border: 1px solid #b7c7ce;
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: white;
    }
    form button {
      margin-top: 16px;
    }
    .result {
      margin-top: 16px;
      padding: 14px;
      border-radius: 8px;
      background: #eef7f4;
      border: 1px solid #b9dad2;
      font-size: 1rem;
    }
    .bar { margin-top: 10px; height: 14px; background: #d9e2e8; border-radius: 999px; overflow: hidden; }
    .bar span { display: block; height: 100%; width: 0; background: var(--warn); transition: width .25s ease; }
    .pill-list { display: flex; flex-wrap: wrap; gap: 8px; padding: 0; list-style: none; }
    .pill-list li { border: 1px solid var(--line); background: white; padding: 7px 10px; border-radius: 999px; }
    .table-wrap { overflow-x: auto; background: white; border: 1px solid var(--line); border-radius: 8px; }
    .data-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    .data-table th, .data-table td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }
    .data-table th { background: var(--soft); color: #12343b; }
    code { background: #e8eeee; padding: 2px 5px; border-radius: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>HR Attrition Prediction</h1>
    <p>Interactive one-page dashboard built from the provided CSV and notebook. It includes EDA charts, model comparison, feature importance, and browser-based prediction.</p>
  </header>
  <nav>
    <div class="inner">
      <a href="#predict">Predict</a>
      <a href="#overview">Overview</a>
      <a href="#eda">EDA Graphs</a>
      <a href="#models">Models</a>
      <a href="#features">Features</a>
      <a href="#tables">Tables</a>
    </div>
  </nav>
  <main>
    <section id="predict">
      <h2>Interactive Attrition Prediction</h2>
      <div class="predictor">
        <p class="note">The form uses the trained Logistic Regression parameters from the notebook, including label encoding and StandardScaler values.</p>
        <form id="prediction-form">
          <div class="form-grid" id="input-grid"></div>
          <button type="submit">Predict Attrition</button>
        </form>
        <div class="result" id="prediction-result">Fill the employee details and click Predict Attrition.</div>
        <div class="bar"><span id="probability-bar"></span></div>
      </div>
    </section>
    <section id="overview">
      <h2>Dataset Overview</h2>
      <div class="grid">
        <div class="stat"><b>__ROWS__</b>Rows in CSV</div>
        <div class="stat"><b>__COLS__</b>Original columns</div>
        <div class="stat"><b>__DUPES__</b>Duplicate rows before cleaning</div>
        <div class="stat"><b>__NUMERIC_COLS__</b>Numeric columns</div>
        <div class="stat"><b>__CATEGORICAL_COLS__</b>Categorical columns</div>
      </div>
      <p class="note">Source files: <code>__DATA_FILE__</code> and <code>__NOTEBOOK_FILE__</code>.</p>
    </section>
    <section id="eda">
      <h2>EDA Graphs</h2>
      <div class="two">
        <div class="chart-card">
          <h3>Main EDA Chart</h3>
          <div class="mini-controls">
            <button type="button" data-chart="target">Attrition</button>
            <button type="button" class="secondary" data-chart="department">Department</button>
            <button type="button" class="secondary" data-chart="jobRole">Job Role</button>
            <button type="button" class="secondary" data-chart="missing">Missing Values</button>
          </div>
          <canvas id="eda-chart" width="920" height="430"></canvas>
        </div>
        <div class="chart-card">
          <h3>Attrition Relationships</h3>
          <div class="mini-controls">
            <button type="button" data-relation="overtimeAttrition">OverTime vs Attrition</button>
            <button type="button" class="secondary" data-relation="ageAttrition">Age Group vs Attrition</button>
            <button type="button" class="secondary" data-relation="incomeByAttrition">Income by Attrition</button>
          </div>
          <canvas id="relation-chart" width="920" height="430"></canvas>
        </div>
      </div>
    </section>
    <section id="models">
      <h2>Model Evaluation Graphs</h2>
      <div class="two">
        <div class="chart-card">
          <h3>Before SMOTE</h3>
          <canvas id="before-chart" width="920" height="430"></canvas>
        </div>
        <div class="chart-card">
          <h3>After SMOTE</h3>
          <canvas id="after-chart" width="920" height="430"></canvas>
        </div>
      </div>
      <p class="note">XGBoost is present in the notebook but skipped here if the package is not installed.</p>
    </section>
    <section id="features">
      <h2>Top Random Forest Features</h2>
      <div class="chart-card">
        <canvas id="importance-chart" width="1100" height="430"></canvas>
      </div>
    </section>
    <section id="tables">
      <h2>Columns Used and Removed</h2>
      <h3>Removed by notebook</h3>
      <ul class="pill-list">__DROPPED_PILLS__</ul>
      <h3>Prediction features after preprocessing</h3>
      <ul class="pill-list">__FEATURE_PILLS__</ul>
    </section>
    <section>
      <h2>EDA Tables</h2>
      <div class="two">
        <div>
          <h3>Target Distribution</h3>
          <div class="table-wrap">__TARGET_TABLE__</div>
        </div>
        <div>
          <h3>Missing Values</h3>
          <div class="table-wrap">__MISSING_TABLE__</div>
        </div>
      </div>
      <h2>EDA Numeric Summary</h2>
      <div class="table-wrap">__NUMERIC_TABLE__</div>
      <p class="note">The notebook imputes numeric missing values with mean and categorical missing values with most frequent value.</p>
    </section>
    <section>
      <h2>Model Evaluation Tables</h2>
      <div class="two">
        <div>
          <h3>Before SMOTE</h3>
          <div class="table-wrap">__BEFORE_TABLE__</div>
        </div>
        <div>
          <h3>After SMOTE</h3>
          <div class="table-wrap">__AFTER_TABLE__</div>
        </div>
      </div>
    </section>
    <section>
      <h2>Feature Importance Table</h2>
      <div class="table-wrap">__IMPORTANCE_TABLE__</div>
    </section>
  </main>
  <script>
    const model = __MODEL_JSON__;
    const charts = __CHART_JSON__;
    const grid = document.getElementById("input-grid");
    const result = document.getElementById("prediction-result");
    const probabilityBar = document.getElementById("probability-bar");

    function fieldId(name) {
      return "field-" + name.replace(/[^a-zA-Z0-9_-]/g, "-");
    }

    model.features.forEach((feature) => {
      const label = document.createElement("label");
      label.textContent = feature.name;
      let input;
      if (feature.type === "categorical") {
        input = document.createElement("select");
        feature.classes.forEach((className, index) => {
          const option = document.createElement("option");
          option.value = String(index);
          option.textContent = className;
          if (className === feature.default) option.selected = true;
          input.appendChild(option);
        });
      } else {
        input = document.createElement("input");
        input.type = "number";
        input.min = String(feature.min);
        input.max = String(feature.max);
        input.step = String(feature.step);
        input.value = String(feature.default);
      }
      input.id = fieldId(feature.name);
      input.name = feature.name;
      label.appendChild(input);
      grid.appendChild(label);
    });

    function sigmoid(value) {
      return 1 / (1 + Math.exp(-value));
    }

    document.getElementById("prediction-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const values = model.features.map((feature) => {
        const raw = document.getElementById(fieldId(feature.name)).value;
        return Number(raw);
      });
      const scaled = values.map((value, index) => {
        const scale = model.scalerScale[index] || 1;
        return (value - model.scalerMean[index]) / scale;
      });
      const score = scaled.reduce((total, value, index) => total + value * model.coef[index], model.intercept);
      const yesProbability = sigmoid(score);
      const predictionIndex = yesProbability >= 0.5 ? 1 : 0;
      const label = model.targetClasses[predictionIndex] || String(predictionIndex);
      result.innerHTML = `<strong>Predicted Attrition:</strong> ${label}<br><strong>Estimated Yes probability:</strong> ${(yesProbability * 100).toFixed(2)}%<br><span class="note">Model used: ${model.modelName}</span>`;
      probabilityBar.style.width = `${Math.max(0, Math.min(100, yesProbability * 100))}%`;
    });

    function resizeCanvas(canvas) {
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth || canvas.width;
      const height = canvas.clientHeight || 320;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx, width, height };
    }

    function drawBar(canvasId, rows, labelKey, valueKey, title, color = "#0f766e") {
      const canvas = document.getElementById(canvasId);
      const { ctx, width, height } = resizeCanvas(canvas);
      ctx.clearRect(0, 0, width, height);
      const pad = { left: 64, right: 20, top: 34, bottom: 86 };
      const values = rows.map(row => Number(row[valueKey]) || 0);
      const maxValue = Math.max(...values, 1);
      const chartW = width - pad.left - pad.right;
      const chartH = height - pad.top - pad.bottom;
      ctx.fillStyle = "#1d252c";
      ctx.font = "700 15px Segoe UI, Arial";
      ctx.fillText(title, pad.left, 20);
      ctx.strokeStyle = "#d7e0e6";
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, pad.top + chartH);
      ctx.lineTo(pad.left + chartW, pad.top + chartH);
      ctx.stroke();
      rows.forEach((row, index) => {
        const gap = 10;
        const barW = Math.max(18, (chartW / rows.length) - gap);
        const x = pad.left + index * (chartW / rows.length) + gap / 2;
        const barH = (Number(row[valueKey]) / maxValue) * chartH;
        const y = pad.top + chartH - barH;
        ctx.fillStyle = color;
        ctx.fillRect(x, y, barW, barH);
        ctx.fillStyle = "#1d252c";
        ctx.font = "12px Segoe UI, Arial";
        ctx.fillText(String(row[valueKey]), x, Math.max(y - 6, pad.top + 12));
        ctx.save();
        ctx.translate(x + 4, pad.top + chartH + 16);
        ctx.rotate(-0.72);
        ctx.fillText(String(row[labelKey]).slice(0, 20), 0, 0);
        ctx.restore();
      });
    }

    function drawGrouped(canvasId, rows, labelKey, keys, title) {
      const canvas = document.getElementById(canvasId);
      const { ctx, width, height } = resizeCanvas(canvas);
      ctx.clearRect(0, 0, width, height);
      const pad = { left: 64, right: 22, top: 42, bottom: 86 };
      const colors = ["#0f766e", "#b45309", "#426b8c"];
      const maxValue = Math.max(...rows.flatMap(row => keys.map(key => Number(row[key]) || 0)), 1);
      const chartW = width - pad.left - pad.right;
      const chartH = height - pad.top - pad.bottom;
      ctx.fillStyle = "#1d252c";
      ctx.font = "700 15px Segoe UI, Arial";
      ctx.fillText(title, pad.left, 20);
      keys.forEach((key, idx) => {
        ctx.fillStyle = colors[idx % colors.length];
        ctx.fillRect(pad.left + idx * 92, 28, 12, 12);
        ctx.fillStyle = "#1d252c";
        ctx.font = "12px Segoe UI, Arial";
        ctx.fillText(key, pad.left + idx * 92 + 16, 39);
      });
      ctx.strokeStyle = "#d7e0e6";
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, pad.top + chartH);
      ctx.lineTo(pad.left + chartW, pad.top + chartH);
      ctx.stroke();
      rows.forEach((row, index) => {
        const groupW = chartW / rows.length;
        const barW = Math.max(12, groupW / (keys.length + 1.5));
        keys.forEach((key, keyIndex) => {
          const value = Number(row[key]) || 0;
          const barH = (value / maxValue) * chartH;
          const x = pad.left + index * groupW + keyIndex * barW + 10;
          const y = pad.top + chartH - barH;
          ctx.fillStyle = colors[keyIndex % colors.length];
          ctx.fillRect(x, y, barW, barH);
        });
        ctx.fillStyle = "#1d252c";
        ctx.font = "12px Segoe UI, Arial";
        ctx.save();
        ctx.translate(pad.left + index * groupW + 10, pad.top + chartH + 16);
        ctx.rotate(-0.72);
        ctx.fillText(String(row[labelKey]).slice(0, 18), 0, 0);
        ctx.restore();
      });
    }

    function drawMetricChart(canvasId, rows, title) {
      const metricRows = rows.map(row => ({ Label: row.Model, Value: Number(row["F1 Score"]) || 0 }));
      drawBar(canvasId, metricRows, "Label", "Value", `${title} - F1 Score`, "#426b8c");
    }

    function setActive(buttons, activeButton) {
      buttons.forEach(button => button.classList.add("secondary"));
      activeButton.classList.remove("secondary");
    }

    document.querySelectorAll("[data-chart]").forEach(button => {
      button.addEventListener("click", () => {
        setActive(document.querySelectorAll("[data-chart]"), button);
        const key = button.dataset.chart;
        if (key === "target") drawBar("eda-chart", charts.target, "Attrition", "Count", "Target Attrition Distribution");
        if (key === "department") drawBar("eda-chart", charts.department, "Department", "Count", "Employees by Department", "#426b8c");
        if (key === "jobRole") drawBar("eda-chart", charts.jobRole, "Job Role", "Count", "Top Job Roles", "#b45309");
        if (key === "missing") drawBar("eda-chart", charts.missing, "Column", "Missing Values", "Missing Values by Column", "#9f3a38");
      });
    });

    document.querySelectorAll("[data-relation]").forEach(button => {
      button.addEventListener("click", () => {
        setActive(document.querySelectorAll("[data-relation]"), button);
        const key = button.dataset.relation;
        if (key === "overtimeAttrition") drawGrouped("relation-chart", charts.overtimeAttrition, "OverTime", ["No", "Yes"], "OverTime vs Attrition");
        if (key === "ageAttrition") drawGrouped("relation-chart", charts.ageAttrition, "Age Group", ["No", "Yes"], "Age Group vs Attrition");
        if (key === "incomeByAttrition") drawBar("relation-chart", charts.incomeByAttrition, "Attrition", "Average Monthly Income", "Average Monthly Income by Attrition", "#426b8c");
      });
    });

    function drawAllCharts() {
      drawBar("eda-chart", charts.target, "Attrition", "Count", "Target Attrition Distribution");
      drawGrouped("relation-chart", charts.overtimeAttrition, "OverTime", ["No", "Yes"], "OverTime vs Attrition");
      drawMetricChart("before-chart", charts.beforeMetrics, "Before SMOTE");
      drawMetricChart("after-chart", charts.afterMetrics, "After SMOTE");
      drawBar("importance-chart", charts.importance, "Feature", "Importance", "Top Random Forest Feature Importance", "#0f766e");
    }

    window.addEventListener("resize", drawAllCharts);
    drawAllCharts();
  </script>
</body>
</html>"""

    replacements = {
        "__MODEL_JSON__": model_json,
        "__CHART_JSON__": chart_json,
        "__ROWS__": str(raw_df.shape[0]),
        "__COLS__": str(raw_df.shape[1]),
        "__DUPES__": str(raw_df.duplicated().sum()),
        "__NUMERIC_COLS__": str(len(numeric_cols)),
        "__CATEGORICAL_COLS__": str(len(categorical_cols)),
        "__DATA_FILE__": html.escape(DATA_PATH.name),
        "__NOTEBOOK_FILE__": html.escape(NOTEBOOK_PATH.name),
        "__DROPPED_PILLS__": dropped_pills,
        "__FEATURE_PILLS__": feature_pills,
        "__TARGET_TABLE__": tables["target"],
        "__MISSING_TABLE__": tables["missing"],
        "__NUMERIC_TABLE__": tables["numeric"],
        "__BEFORE_TABLE__": tables["before"],
        "__AFTER_TABLE__": tables["after"],
        "__IMPORTANCE_TABLE__": tables["importance"],
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def add_pdf_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", color="#12343b")
    y = 0.9
    for line in lines:
        wrapped = textwrap.wrap(line, width=92) or [""]
        for part in wrapped:
            fig.text(0.08, y, part, fontsize=10.5, color="#17202a", va="top")
            y -= 0.024
        y -= 0.012
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            plt.axis("off")
            fig.text(0.08, 0.94, f"{title} continued", fontsize=18, weight="bold", color="#12343b")
            y = 0.9
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf(ns: dict) -> None:
    raw_df = pd.read_csv(DATA_PATH)
    results_df = ns.get("results_df", pd.DataFrame()).round(4)
    smote_results_df = ns.get("smote_results_df", pd.DataFrame()).round(4)
    missing = raw_df.isnull().sum()
    missing_text = ", ".join(f"{col}: {int(val)}" for col, val in missing[missing > 0].items()) or "No missing values found."
    target_text = raw_df["Attrition"].value_counts().to_string()
    feature_text = ", ".join(ns["X"].columns)
    encoded_text = ", ".join(ns["label_encoders"].keys())

    pages = [
        (
            "Project Overview",
            [
                "Project title: HR Employee Attrition Prediction.",
                f"Notebook used: {NOTEBOOK_PATH.name}.",
                f"Dataset used: {DATA_PATH.name}.",
                f"Raw dataset shape: {raw_df.shape[0]} rows and {raw_df.shape[1]} columns.",
                "Goal: predict whether an employee belongs to the Attrition Yes or No class using employee, job, compensation, satisfaction, and tenure-related fields.",
                "Target column: Attrition. The model treats this as a binary classification problem with classes No and Yes.",
                f"Target distribution in the raw CSV: {target_text}",
                "The project now has two user interfaces. app.py is the Streamlit Python app that reuses the notebook pipeline. index.html is a one-page browser dashboard with EDA charts and a JavaScript prediction form based on the trained Logistic Regression parameters.",
            ],
        ),
        (
            "Notebook Code Flow",
            [
                "Cell 1 imports basic Kaggle environment helpers. It is not needed for local deployment, so notebook_bridge.py skips it when running the notebook for the apps.",
                "The import cell loads pandas and numpy for data handling, matplotlib and seaborn for plots, scikit-learn utilities for splitting/scaling/models/metrics, and imblearn.SMOTE for class balancing.",
                "The loading cell reads the HR attrition CSV into df and prints the first rows, dataset shape, info, null counts, and descriptive statistics. This gives the first understanding of row count, column types, and missing data.",
                f"Duplicate handling: df.duplicated().sum() is checked, then df.drop_duplicates(inplace=True) removes duplicate rows. Raw duplicate count found: {raw_df.duplicated().sum()}.",
                f"Missing value handling: numeric columns are imputed with mean and categorical columns with most frequent value. Missing values found in raw data: {missing_text}",
                f"Label encoding: object columns are converted into numeric values using LabelEncoder. Encoded columns include: {encoded_text}. This is required because the selected machine learning algorithms expect numeric input.",
                "Constant and ID column detection is done by checking the number of unique values in each column. A column with one unique value is constant. A column with unique values equal to total rows is ID-like.",
                "Dropped columns: EmployeeCount, EmployeeNumber, Over18, and StandardHours. EmployeeCount, Over18, and StandardHours do not vary meaningfully, while EmployeeNumber is an identifier and should not guide prediction.",
                "EDA plots in the notebook include a correlation heatmap, decision tree visualization, Random Forest feature importance, model metric comparison, and before/after SMOTE comparison.",
            ],
        ),
        (
            "Feature Engineering",
            [
                "Feature selection is simple and transparent: X is created by dropping Attrition from df, and y is created from df['Attrition'].",
                f"Final model feature count after preprocessing: {ns['X'].shape[1]}. Features used: {feature_text}",
                "train_test_split(X, y, test_size=0.2, random_state=42) creates an 80 percent training set and 20 percent test set. random_state=42 makes the split reproducible.",
                "StandardScaler is fitted only on X_train and then applied to X_train and X_test. This avoids data leakage because the test set is not used to learn the scaling parameters.",
                "Scaling is especially important for Logistic Regression and KNN because these models are affected by feature magnitude. It also keeps model comparisons consistent.",
            ],
        ),
        (
            "Models and Evaluation",
            [
                "Logistic Regression is used as a linear baseline model. It learns weighted relationships between scaled employee features and the probability of attrition.",
                "Random Forest is used because it can capture non-linear relationships and interactions between features. It also provides feature importance values used in the dashboard.",
                "CART Decision Tree is used as an interpretable tree model with criterion='gini' and max_depth=5. The max depth limits overfitting and keeps the tree readable.",
                "KNN is used as a distance-based classifier with n_neighbors=5. Since KNN depends on distances, scaling is necessary before training.",
                "XGBoost code exists in the notebook. The local environment used here does not have xgboost installed, so notebook_bridge.py skips only the XGBoost cells and keeps the rest of the workflow working.",
                "The evaluate_model function calculates Accuracy, Precision, Recall, F1 Score, ROC AUC, classification report, and confusion matrix. These metrics are more informative than accuracy alone because attrition classes are imbalanced.",
                "Accuracy measures total correct predictions. Precision measures how many predicted attrition cases were actually attrition. Recall measures how many actual attrition cases were found. F1 balances precision and recall. ROC AUC summarizes class separation.",
                "SMOTE is applied only to training data after scaling. This creates synthetic minority-class samples in the training set while keeping the test set untouched for fair evaluation.",
                "Before SMOTE results:",
                results_df.to_string(index=False) if not results_df.empty else "No before-SMOTE results were generated.",
                "After SMOTE results:",
                smote_results_df.to_string(index=False) if not smote_results_df.empty else "No after-SMOTE results were generated.",
            ],
        ),
        (
            "Streamlit App Code",
            [
                "app.py is the Python Streamlit frontend. It does not rewrite the complete ML code. Instead, it calls load_notebook_namespace() from notebook_bridge.py.",
                "st.set_page_config sets the browser title, icon, and layout. st.cache_resource caches notebook execution so the notebook does not retrain on every small UI refresh.",
                "The app reads raw_df, X, scaler, label_encoders, and trained models from the notebook namespace. This keeps the app connected to the original notebook logic.",
                "The model dictionary exposes Random Forest, Logistic Regression, CART Decision Tree, KNN, and their SMOTE versions when available.",
                "For each feature in X.columns, the app creates a Streamlit input widget. Categorical columns become selectbox widgets using the original LabelEncoder classes. Numeric columns become number_input widgets using actual min, max, and median values from the CSV.",
                "When the user clicks Predict Attrition, the app builds a one-row DataFrame in the same feature order as training. Categorical values are transformed through the same label encoders, then scaler.transform() applies the training scaler.",
                "The selected trained model predicts the class. If predict_proba exists, the app also displays class probabilities as a bar chart.",
            ],
        ),
        (
            "Notebook Bridge Code",
            [
                "notebook_bridge.py is the connector between the existing ipynb and the new apps.",
                "BASE_DIR, NOTEBOOK_PATH, and DATA_PATH locate the notebook and CSV relative to the project folder, so the project can run from this directory without using the old Kaggle or D:/ML Project path.",
                "_prepare_source() modifies notebook cells at runtime. It replaces the old CSV path with the local CSV path, skips Kaggle input listing, skips heavy notebook-only plot cells, and skips XGBoost cells when xgboost is unavailable.",
                "load_notebook_namespace() reads the notebook with nbformat, executes each usable code cell, suppresses print/display output, closes matplotlib figures, and returns the namespace dictionary containing trained variables.",
                "This approach satisfies the requirement to connect the ipynb instead of rewriting the full Streamlit ML pipeline.",
            ],
        ),
        (
            "HTML Frontend Code",
            [
                "index.html is a one-page dashboard that can open directly in a browser. It does not require Streamlit, Flask, internet, or external JavaScript libraries.",
                "The top part contains navigation links for Predict, Overview, EDA Graphs, Models, Features, and Tables.",
                "The EDA chart section uses Canvas. Buttons switch between Attrition distribution, Department distribution, Job Role distribution, and Missing Values.",
                "The relationship chart section shows OverTime vs Attrition, Age Group vs Attrition, and Average Monthly Income by Attrition using data computed from the actual CSV.",
                "The model section draws before-SMOTE and after-SMOTE F1 score charts from the notebook result tables.",
                "The feature section draws top Random Forest feature importance from rf.feature_importances_ and X.columns.",
                "The prediction form in HTML uses the trained Logistic Regression model. generate_frontend_and_docs.py exports the model coefficients, intercept, scaler means, scaler scales, feature metadata, and target classes into JavaScript.",
                "When Predict Attrition is clicked, JavaScript collects values, label-encodes categorical selections by their class index, scales each feature using (value - scalerMean) / scalerScale, calculates the Logistic Regression score, applies sigmoid, and returns No or Yes based on a 0.5 threshold.",
            ],
        ),
        (
            "Generator Script Code",
            [
                "generate_frontend_and_docs.py is the reproducibility script. Running it regenerates index.html and HR_Attrition_Code_Documentation.pdf from the current notebook and CSV.",
                "build_html() reads dataset summaries, missing values, model metrics, feature importance, chart payloads, and Logistic Regression parameters, then writes a complete self-contained HTML file.",
                "build_pdf() writes this documentation using matplotlib.backends.backend_pdf.PdfPages. This avoids requiring extra PDF libraries.",
                "add_pdf_page() wraps long text onto PDF pages and automatically continues on a new page when needed.",
                "If the dataset or notebook changes later, running python generate_frontend_and_docs.py updates both the dashboard and the documentation.",
            ],
        ),
        (
            "Project Files",
            [
                "finalcodemlproject.ipynb: original ML notebook containing data loading, cleaning, preprocessing, training, evaluation, SMOTE, and visualizations.",
                "Imperfect_HR_Attrition (1) (1).csv: source HR attrition dataset.",
                "notebook_bridge.py: executes the notebook safely for local app use and returns trained variables.",
                "app.py: Streamlit app for interactive prediction using notebook-trained models.",
                "index.html: standalone interactive dashboard with EDA graphs and browser prediction.",
                "generate_frontend_and_docs.py: regenerates the HTML frontend and documentation PDF.",
                "HR_Attrition_Code_Documentation.pdf: detailed code documentation and team division.",
            ],
        ),
        (
            "Three Member Work Division",
            [
                "Member 1: Data Understanding and EDA. Responsibilities: loading the CSV in the notebook, checking df.head(), df.shape, df.info(), null values, duplicate rows, and df.describe(). This member also prepared EDA interpretation using target distribution, department distribution, job role distribution, overtime relation, age group relation, income comparison, correlation heatmap, and missing-value analysis.",
                "Member 1 output: dataset understanding section, EDA graph section in HTML, and explanation of what the data contains before modelling.",
                "Member 2: Preprocessing and Model Building. Responsibilities: dropping duplicate rows, imputing numeric columns with mean, imputing categorical columns with most frequent value, label encoding object columns, identifying constant/ID-like columns, dropping EmployeeCount, EmployeeNumber, Over18, and StandardHours, splitting X and y, applying StandardScaler, and training Logistic Regression, Random Forest, CART Decision Tree, KNN, and optional XGBoost.",
                "Member 2 output: cleaned modelling dataset, trained classifiers, saved notebook variables, and Random Forest feature importance.",
                "Member 3: Evaluation, UI, Deployment, and Documentation. Responsibilities: writing evaluation functions, generating confusion matrices, classification reports, before/after SMOTE metric tables, SMOTE balancing, Streamlit app, notebook bridge, interactive HTML dashboard, JavaScript prediction logic, and this PDF documentation.",
                "Member 3 output: app.py, notebook_bridge.py, index.html, generate_frontend_and_docs.py, and HR_Attrition_Code_Documentation.pdf.",
                "This division is written according to the completed project work and can be used in presentation or viva to explain individual contributions clearly.",
            ],
        ),
    ]

    with PdfPages(PDF_PATH) as pdf:
        for title, lines in pages:
            add_pdf_page(pdf, title, lines)


def main() -> None:
    ns = load_notebook_namespace()
    HTML_PATH.write_text(build_html(ns), encoding="utf-8")
    build_pdf(ns)
    print(f"Created {HTML_PATH}")
    print(f"Created {PDF_PATH}")


if __name__ == "__main__":
    main()
