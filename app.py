"""Heart Disease Detection – Streamlit app. Uses the best model chosen by pipeline.py."""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from pipeline import run_pipeline, load_and_validate, CONTINUOUS

BUNDLE = os.path.join("artifacts", "model_bundle.joblib")
DATA_FILE = "Cardiovascular_Disease_Dataset.csv"
LABEL = {0: "No Disease", 1: "Disease"}

# ───────────── Data dictionary (names, units, value meanings) ─────────────
FEATURES = {  # column: (display name, description)
    "age": ("Age", "In years"),
    "gender": ("Gender", "Binary"),
    "chestpain": ("Chest pain type", "Nominal"),
    "restingBP": ("Resting blood pressure", "mm Hg, 94–200"),
    "serumcholestrol": ("Serum cholesterol", "mg/dl, 126–564"),
    "fastingbloodsugar": ("Fasting blood sugar > 120 mg/dl", "Binary"),
    "restingrelectro": ("Resting electrocardiogram", "Nominal"),
    "maxheartrate": ("Maximum heart rate achieved", "71–202"),
    "exerciseangia": ("Exercise-induced angina", "Binary"),
    "oldpeak": ("Oldpeak (ST depression)", "0–6.2"),
    "slope": ("Slope of peak exercise ST segment", "Nominal"),
    "noofmajorvessels": ("Number of major vessels", "0–3"),
    "target": ("Heart disease (target)", "Binary"),
}
VALUE_LABELS = {
    "gender": {0: "Female", 1: "Male"},
    "fastingbloodsugar": {0: "No (≤ 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
    "chestpain": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
    "restingrelectro": {0: "Normal", 1: "ST-T wave abnormality",
                        2: "Probable/definite left ventricular hypertrophy"},
    "exerciseangia": {0: "No", 1: "Yes"},
    "slope": {0: "Not specified (0)", 1: "Upsloping", 2: "Flat", 3: "Downsloping"},
}
REQUIRED_COLUMNS = list(FEATURES)  # exact names the CSV must contain (patientid optional)


def fname(col):
    return FEATURES.get(col, (col,))[0]


def vlabel(col, v):
    return f"{VALUE_LABELS.get(col, {}).get(v, v)}"


def column_guide():
    rows = []
    for c, (name, note) in FEATURES.items():
        vals = VALUE_LABELS.get(c)
        meaning = "; ".join(f"{k} = {v}" for k, v in vals.items()) if vals else note
        rows.append({"Column name (exact)": c, "Meaning": name, "Values / range": meaning})
    return pd.DataFrame(rows)


st.set_page_config(page_title="Heart Disease Detection", page_icon="❤️", layout="wide")


# ───────────── Light / dark mode (charts follow the app theme) ─────────────
def detect_dark():
    """True if the user's Streamlit theme is dark. Needs Streamlit ≥ 1.46; falls back to config, then dark."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("dark", "light"):
            return base == "dark"
    except Exception:
        pass
    return True


IS_DARK = detect_dark()
if IS_DARK:
    PLOT_TEMPLATE = "plotly_dark"
    COLORS = {"No Disease": "#2DD4BF", "Disease": "#F43F5E"}          # teal = healthy, coral-red = disease
    MODEL_COLORS = ["#38BDF8", "#FBBF24", "#A78BFA", "#34D399", "#FB7185", "#F97316"]
    ACCENT, HIGHLIGHT, SCALE_SEQ = "#38BDF8", "#0f766e", "Teal"
else:
    PLOT_TEMPLATE = "plotly_white"
    COLORS = {"No Disease": "#0D9488", "Disease": "#E11D48"}
    MODEL_COLORS = ["#0284C7", "#D97706", "#7C3AED", "#059669", "#E11D48", "#EA580C"]
    ACCENT, HIGHLIGHT, SCALE_SEQ = "#0284C7", "#99f6e4", "Teal"


def stretch(fn, obj, **kw):
    """Full-width rendering that works on both old and new Streamlit versions."""
    try:
        return fn(obj, width="stretch", **kw)
    except (TypeError, st.errors.StreamlitAPIException):
        return fn(obj, use_container_width=True, **kw)


def show(fig, container=st, **layout):
    """Style a Plotly figure for the current light/dark mode and render it full-width."""
    fig.update_layout(template=PLOT_TEMPLATE, colorway=MODEL_COLORS,
                      margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(**layout)
    stretch(container.plotly_chart, fig, theme=None)  # theme=None -> our template controls colours


def table(df_):
    stretch(st.dataframe, df_)


@st.cache_resource(show_spinner="Loading model…")
def load_bundle(df: pd.DataFrame, use_saved: bool):
    """Use artifacts from `python pipeline.py` if present; otherwise run the pipeline now."""
    if use_saved and os.path.exists(BUNDLE):
        return joblib.load(BUNDLE)
    return run_pipeline(df, out_dir=None, verbose=False)


# ───────────── Load data ─────────────
st.sidebar.title("❤️ Heart Disease Detection")
raw = pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else None
upload = st.sidebar.file_uploader("Dataset CSV (optional if file is next to app.py)", type="csv")
if upload is not None:
    raw = pd.read_csv(upload)

if raw is None:
    st.title("Heart Disease Detection")
    st.info(f"Place **{DATA_FILE}** next to `app.py`, or upload it from the sidebar.")
    st.subheader("Required column names")
    st.caption("Your CSV must use exactly these column names (case-insensitive; `patientid` is optional).")
    stretch(st.dataframe, column_guide())
    st.stop()

try:
    df = load_and_validate(raw)
except ValueError as e:
    st.error(str(e))
    st.subheader("Required column names")
    st.caption("Your CSV must use exactly these column names (case-insensitive; `patientid` is optional).")
    stretch(st.dataframe, column_guide())
    st.stop()

art = load_bundle(df, use_saved=upload is None)
best = art["best_model"]
best_pipe = art["models"][best]
res = art["results"]

page = st.sidebar.radio("Navigate", ["🩺 Predict", "📊 Data Explorer", "🏆 Model Performance"])
st.sidebar.success(f"Best model: **{best}**")
st.sidebar.caption(f"{len(df):,} patients · {df.shape[1]-1} features")
with st.sidebar.expander("📋 Dataset column guide"):
    st.caption("CSV columns must have these exact names:")
    st.dataframe(column_guide(), hide_index=True)

# ───────────── Predict (best model only) ─────────────
if page == "🩺 Predict":
    st.title("🩺 Heart Disease Risk Prediction")
    st.caption(f"Prediction uses the best model: **{best}** "
               f"(CV ROC-AUC {res.loc[best,'CV ROC-AUC (mean)']:.3f} · test ROC-AUC {res.loc[best,'ROC-AUC']:.3f}).")

    def rng(col, default, lo=None, hi=None):
        lo = float(df[col].min()) if lo is None else lo
        hi = float(df[col].max()) if hi is None else hi
        return lo, hi, float(np.clip(default, lo, hi))

    def choice(container, col):
        """Selectbox showing meaningful labels; also offers any extra codes found in the data."""
        opts = list(VALUE_LABELS[col])
        opts += [v for v in sorted(df[col].unique()) if v not in opts]
        return container.selectbox(fname(col), opts, format_func=lambda v: vlabel(col, v))

    c1, c2, c3 = st.columns(3)
    with c1:
        lo, hi, d = rng("age", 50)
        age = c1.slider(f"{fname('age')} (years)", int(lo), int(hi), int(d))
        gender = choice(c1, "gender")
        chestpain = choice(c1, "chestpain")
        lo, hi, d = rng("restingBP", 130, 94, 200)
        bp = c1.slider(f"{fname('restingBP')} (mm Hg)", int(lo), int(hi), int(d))
        lo, hi, d = rng("serumcholestrol", 240, 126, 564)
        chol = c1.slider(f"{fname('serumcholestrol')} (mg/dl)", int(lo), int(hi), int(d))
    with c2:
        fbs = choice(c2, "fastingbloodsugar")
        ecg = choice(c2, "restingrelectro")
        lo, hi, d = rng("maxheartrate", 150, 71, 202)
        mhr = c2.slider(fname("maxheartrate"), int(lo), int(hi), int(d))
        angina = choice(c2, "exerciseangia")
    with c3:
        lo, hi, d = rng("oldpeak", 1.0, 0.0, 6.2)
        oldpeak = c3.slider(fname("oldpeak"), lo, hi, d, step=0.1)
        slope = choice(c3, "slope")
        vessels = c3.selectbox(fname("noofmajorvessels"), [0, 1, 2, 3])

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([dict(age=age, gender=gender, chestpain=chestpain, restingBP=bp,
                                 serumcholestrol=chol, fastingbloodsugar=fbs, restingrelectro=ecg,
                                 maxheartrate=mhr, exerciseangia=angina, oldpeak=oldpeak,
                                 slope=slope, noofmajorvessels=vessels)])[art["features"]]
        prob = float(best_pipe.predict_proba(row)[0, 1])
        a, b = st.columns([1, 2])
        a.metric("Disease probability", f"{prob:.1%}")
        if prob >= 0.5:
            b.error("⚠️ Model predicts **heart disease likely**. Consult a clinician.")
        else:
            b.success("✅ Model predicts **low likelihood** of heart disease.")
        b.progress(prob)
    st.warning("Educational demo only — not a medical device or a substitute for professional diagnosis.")

# ───────────── Data Explorer ─────────────
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    d = df.copy()
    d["Class"] = d["target"].map(LABEL)
    t1, t2, t3, t4 = st.tabs(["Overview", "Distributions", "Categorical", "Correlation"])
    with t1:
        counts = d["Class"].value_counts()
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(d):,}"); b.metric("No Disease", int(counts.get("No Disease", 0)))
        c.metric("Disease", int(counts.get("Disease", 0)))
        show(px.bar(counts.reset_index(), x="Class", y="count", color="Class",
                    color_discrete_map=COLORS, title="Class balance"))
        table(d.drop(columns="Class").head(50))
        table(df[CONTINUOUS].describe().round(3).T)
    with t2:
        feat = st.selectbox(
            "Continuous feature",
            CONTINUOUS,
            format_func=fname
        )

        fig = px.histogram(
            d,
            x=feat,
            color="Class",
            nbins=30,
            barmode="overlay",
            opacity=0.55,
            histnorm="probability density",
            color_discrete_map=COLORS,
            labels={feat: fname(feat)},
            title=f"{fname(feat)} Distribution by Class"
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title=fname(feat),
            yaxis_title="Probability Density",
            legend_title="Class"
        )

        show(fig)
    with t3:
        cats = [c for c in df.columns if c not in CONTINUOUS + ["target"]]
        feat = st.selectbox("Categorical feature", cats, format_func=fname)
        g = d.groupby([feat, "Class"]).size().reset_index(name="count")
        g[fname(feat)] = g[feat].map(lambda v: vlabel(feat, v))
        show(px.bar(g, x=fname(feat), y="count", color="Class", barmode="group",
                    color_discrete_map=COLORS, title=f"{fname(feat)} by class"))
    with t4:
        corr = df.corr().round(2).rename(index=fname, columns=fname)
        show(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, color_continuous_midpoint=0,
                       aspect="auto", title="Feature correlation").update_layout(height=650))

# ───────────── Model Performance ─────────────
else:
    st.title("🏆 Model Performance")
    st.success(f"Best model (selected by 5-fold CV ROC-AUC): **{best}** — hold-out AUC "
               f"{res.loc[best,'ROC-AUC']:.3f}, F1 {res.loc[best,'F1 Score']:.3f}")
    table(res.style.format("{:.4f}").highlight_max(axis=0, color=HIGHLIGHT))

    X_te, y_te = art["X_te"], art["y_te"]
    t1, t2, t3= st.tabs(["ROC curves", "Confusion matrices", "Cross-validation"])
    with t1:
        fig = go.Figure()
        for i, (n, m) in enumerate(art["models"].items()):
            pr = m.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, pr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{n} ({roc_auc_score(y_te, pr):.3f})",
                                     line=dict(width=4 if n == best else 2, color=MODEL_COLORS[i % len(MODEL_COLORS)])))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                 line=dict(dash="dash", color="grey")))
        fig.update_layout(
            title="ROC curves (test set) — Top-Left Corner Focus",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[-0.01, 0.25]),
            yaxis=dict(range=[0.75, 1.01]),
            height=550
        )
        show(fig)
    with t2:
        names = list(art["models"])
        for i in range(0, len(names), 3):
            cols = st.columns(3)
            for col, n in zip(cols, names[i:i + 3]):
                cm = confusion_matrix(y_te, art["models"][n].predict(X_te))
                f = px.imshow(cm, text_auto=True, color_continuous_scale=SCALE_SEQ, x=["Pred No", "Pred Yes"],
                              y=["True No", "True Yes"], title=n + (" ⭐" if n == best else ""))
                show(f, col, coloraxis_showscale=False, height=330)
    with t3:
        cvdf = pd.DataFrame([(n, v) for n, vals in art["cv"].items() for v in vals], columns=["Model", "ROC-AUC"])
        show(px.box(cvdf, x="Model", y="ROC-AUC", color="Model", color_discrete_sequence=MODEL_COLORS, points="all", title="5-fold CV ROC-AUC"))
   