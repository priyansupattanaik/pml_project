from __future__ import annotations

import pandas as pd
import streamlit as st

from notebook_bridge import DATA_PATH, NOTEBOOK_PATH, load_notebook_namespace


st.set_page_config(page_title="HR Attrition Predictor", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading notebook and training models...")
def get_project_namespace():
    return load_notebook_namespace()


def decode_prediction(value: int, label_encoders: dict) -> str:
    encoder = label_encoders.get("Attrition")
    if encoder is None:
        return str(value)
    return str(encoder.inverse_transform([int(value)])[0])


ns = get_project_namespace()
raw_df: pd.DataFrame = ns["raw_df"]
X: pd.DataFrame = ns["X"]
scaler = ns["scaler"]
label_encoders = ns["label_encoders"]

models = {
    "Random Forest": ns.get("rf"),
    "Logistic Regression": ns.get("lr"),
    "CART Decision Tree": ns.get("dt"),
    "KNN": ns.get("knn"),
    "Random Forest + SMOTE": ns.get("rf_smote"),
    "Logistic Regression + SMOTE": ns.get("lr_smote"),
    "Decision Tree + SMOTE": ns.get("dt_smote"),
    "KNN + SMOTE": ns.get("knn_smote"),
}
models = {name: model for name, model in models.items() if model is not None}

st.title("HR Employee Attrition Prediction")
st.caption(f"Prediction app connected to `{NOTEBOOK_PATH.name}` and `{DATA_PATH.name}`.")

left, right = st.columns([0.65, 0.35])

with right:
    st.subheader("Project Snapshot")
    st.metric("Rows", f"{raw_df.shape[0]:,}")
    st.metric("Columns", f"{raw_df.shape[1]:,}")
    st.metric("Target", "Attrition")
    if not ns.get("xgboost_available", False):
        st.info("XGBoost cells are skipped because `xgboost` is not installed in this environment.")

    results_df = ns.get("results_df")
    smote_results_df = ns.get("smote_results_df")
    if results_df is not None:
        st.write("Before SMOTE")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    if smote_results_df is not None:
        st.write("After SMOTE")
        st.dataframe(smote_results_df, use_container_width=True, hide_index=True)

with left:
    st.subheader("Enter Employee Details")
    result_box = st.container()

    with st.form("attrition_prediction_form"):
        selected_model = st.selectbox("Model", list(models.keys()), index=0)

        form_values = {}
        cols = st.columns(3)
        for idx, feature in enumerate(X.columns):
            source_series = raw_df[feature] if feature in raw_df.columns else X[feature]
            with cols[idx % 3]:
                if feature in label_encoders:
                    classes = [str(item) for item in label_encoders[feature].classes_]
                    mode_value = str(source_series.mode(dropna=True).iloc[0]) if not source_series.mode(dropna=True).empty else classes[0]
                    default_index = classes.index(mode_value) if mode_value in classes else 0
                    form_values[feature] = st.selectbox(feature, classes, index=default_index)
                else:
                    numeric_series = pd.to_numeric(source_series, errors="coerce")
                    min_value = float(numeric_series.min())
                    max_value = float(numeric_series.max())
                    median_value = float(numeric_series.median())
                    step = 1.0 if (numeric_series.dropna() % 1 == 0).all() else 0.1
                    form_values[feature] = st.number_input(
                        feature,
                        min_value=min_value,
                        max_value=max_value,
                        value=median_value,
                        step=step,
                    )

        submitted = st.form_submit_button("Predict Attrition", type="primary")

    if submitted:
        try:
            input_df = pd.DataFrame([form_values], columns=X.columns)
            for feature, encoder in label_encoders.items():
                if feature in input_df.columns:
                    input_df[feature] = encoder.transform(input_df[feature].astype(str))

            input_scaled = scaler.transform(input_df)
            model = models[selected_model]
            prediction = int(model.predict(input_scaled)[0])
            label = decode_prediction(prediction, label_encoders)

            with result_box:
                st.success(f"Predicted Attrition: {label}")
                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(input_scaled)[0]
                    class_names = [decode_prediction(i, label_encoders) for i in range(len(probability))]
                    st.bar_chart(pd.DataFrame({"Probability": probability}, index=class_names))
        except Exception as exc:
            with result_box:
                st.error(f"Prediction failed: {exc}")
