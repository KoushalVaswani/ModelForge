import streamlit as st
import pandas as pd
from preprocessing import create_preprocessor
import base64
from pathlib import Path

from model_registry import (
    get_classification_models,
    get_regression_models,
    get_classification_param_grids,
    get_regression_param_grids
)

from trainer import (
    train_classification_models,
    train_regression_models,
)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="ModelForge",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# MODELFORGE UI STYLING
# =====================================================

background_path = Path("ModelForge.png")

with open(background_path, "rb") as image_file:
    encoded_image = base64.b64encode(
        image_file.read()
    ).decode()

st.markdown(
    f"""
    <style>

        /* ================================
           FULL BACKGROUND
           ================================ */

        .stApp {{
            background:
                linear-gradient(
                    rgba(5, 7, 13, 0.55),
                    rgba(5, 7, 13, 0.55)
                ),
                url("data:image/png;base64,{encoded_image}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}


        /* ================================
           STREAMLIT MAIN AREA
           ================================ */

        [data-testid="stAppViewContainer"] {{
            background: transparent !important;
        }}

        [data-testid="stMain"] {{
            background: transparent !important;
        }}


        /* ================================
           HEADER
           ================================ */

        [data-testid="stHeader"] {{
            background: rgba(5, 7, 13, 0.75) !important;
        }}

        [data-testid="stDecoration"] {{
            display: none;
        }}


        /* ================================
           MAIN CONTENT
           ================================ */

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}


        /* ================================
           TITLE
           ================================ */

        .main-title {{
            font-size: 3.2rem;
            font-weight: 850;
            letter-spacing: -1px;
            margin-bottom: 0.1rem;
            background: linear-gradient(90deg, #ffffff, #8ab4ff, #b58cff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            color: #a1a9b8;
            font-size: 1.05rem;
            margin-top: 0.2rem;
            margin-bottom: 1rem
        }}

        .header-divider {{
            height: 1px;
            margin: 0.8rem 0 2rem 0;
            background: linear-gradient(
                90deg,
                rgba(88, 166, 255, 0.6),
                rgba(139, 92, 246, 0.3),
                transparent
            );
        }}


        /* ================================
           HEADINGS
           ================================ */

        h1, h2, h3 {{
            font-weight: 700 !important;
        }}


        /* ================================
           METRIC CARDS
           ================================ */

        div[data-testid="stMetric"] {{
            background: rgba(30,36,46,0.85);
            border: 1px solid rgba(255, 255, 255, 0.14);
            padding: 18px;
            border-radius: 14px;
            backdrop-filter: blur(10px);
        }}

        div[data-testid="stMetricLabel"] {{
            color: #a8b0bd;
        }}


        /* ================================
           BUTTONS
           ================================ */

        .stButton > button {{
            width: 100%;
            border-radius: 10px;
            padding: 0.65rem 1rem;
            font-weight: 700;
            border: 1px solid #3a414b;
            transition: 0.2s;
        }}

        .stButton > button:hover {{
            border-color: #58a6ff;
            transform: translateY(-1px);
        }}


        /* ================================
           FILE UPLOADER
           ================================ */

        div[data-testid="stFileUploader"] {{
            background: rgba(22, 27, 34, 0.92);
            border: 1px dashed #30363d;
            border-radius: 14px;
            padding: 10px;
            backdrop-filter: blur(8px);
        }}


        /* ================================
           DATAFRAMES
           ================================ */

        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
        }}


        /* ================================
           ALERTS
           ================================ */

        div[data-testid="stAlert"] {{
            border-radius: 10px;
        }}

    </style>
    """,
    unsafe_allow_html=True
)
def optimize_models(
    X,
    y,
    preprocessor,
    models,
    param_grids,
    scoring
):
    results = []

    for name, model in models.items():

        if name not in param_grids:
            continue

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=3,
            cv=3,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            refit=True
        )

        search.fit(X, y)

        results.append({
            "Model": name,
            "Best Score": search.best_score_,
            "Best Parameters": search.best_params_,
            "Best Estimator": search.best_estimator_
        })

    return results

def evaluate_best_model(X, y, best_estimator, problem_type):

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )
    import numpy as np

    if problem_type == "Classification":

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    best_estimator.fit(X_train, y_train)

    predictions = best_estimator.predict(X_test)

    if problem_type == "Classification":

        return {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        }

    else:

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        return {
            "MAE": mean_absolute_error(
                y_test,
                predictions
            ),
            "RMSE": rmse,
            "R² Score": r2_score(
                y_test,
                predictions
            )
        }



st.set_page_config(
    page_title="ModelForge",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    '<div class="main-title">⚒️ ModelForge</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Automated Machine Learning Model Selection & Optimization'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="header-divider"></div>',
    unsafe_allow_html=True
)

st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help="Upload a clean CSV file containing your features and target column."
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(
        f"✅ Dataset loaded successfully — "
    )

    # =========================================================
    # DATASET OVERVIEW
    # =========================================================

    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            df.duplicated().sum()
        )

    # Dataset Preview
    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # =========================================================
    # COLUMN INFORMATION
    # =========================================================

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

    # =========================================================
    # TARGET SELECTION
    # =========================================================

    st.header("🎯 Target Selection")

    target_column = st.selectbox(
        "Select the target column you want to predict:",
        df.columns
    )

    if target_column:

        X = df.drop(columns=[target_column])
        y = df[target_column]

        st.write(
            f"**Target:** `{target_column}`"
        )

        # =====================================================
        # PROBLEM TYPE DETECTION
        # =====================================================

        if pd.api.types.is_numeric_dtype(y):

            if y.nunique() <= 10:
                problem_type = "Classification"
            else:
                problem_type = "Regression"

        else:
            problem_type = "Classification"

        st.success(
            f"Detected Problem Type: **{problem_type}**"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Features",
                X.shape[1]
            )

        with col2:
            st.metric(
                "Target Unique Values",
                y.nunique()
            )

        # =====================================================
        # AUTOMATIC PREPROCESSING
        # =====================================================

        preprocessor, numerical_columns, categorical_columns = (
            create_preprocessor(X)
        )

        st.header("⚙️ Automatic Preprocessing")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Numerical Features**")

            if numerical_columns:
                st.write(numerical_columns)
            else:
                st.write("No numerical features")

        with col2:

            st.write("**Categorical Features**")

            if categorical_columns:
                st.write(categorical_columns)
            else:
                st.write("No categorical features")

        st.success(
            "Preprocessing pipeline created successfully!"
        )

        # =====================================================
        # MODEL ARENA
        # =====================================================

        st.header("⚔️ Model Arena")

        if st.button("🚀 Train & Compare Models"):

            with st.spinner(
                "Training models..."
            ):

                if problem_type == "Classification":

                    models = get_classification_models()

                    results = train_classification_models(
                        X,
                        y,
                        preprocessor,
                        models
                    )

                    results_df = pd.DataFrame(results)

                    results_df = results_df.sort_values(
                        by="F1 Score",
                        ascending=False
                    )

                else:

                    models = get_regression_models()

                    results = train_regression_models(
                        X,
                        y,
                        preprocessor,
                        models
                    )

                    results_df = pd.DataFrame(results)

                    results_df = results_df.sort_values(
                        by="R2 Score",
                        ascending=False
                    )

                st.subheader(
                    "🏆 Model Leaderboard"
                )
                st.caption(
                    "Baseline performance of all candidate models on the test split."
                )
                best_baseline = results_df.iloc[0]

                st.info(
                    f"💡 Best baseline model: **{best_baseline['Model']}**"
                )

                st.dataframe(
                    results_df,
                    use_container_width=True
                )
                st.subheader("📊 Model Performance Comparison")

                if problem_type == "Classification":

                    chart_df = results_df[
                        ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
                    ].copy()

                    chart_df = chart_df.set_index("Model")

                    st.bar_chart(
                        chart_df,
                        use_container_width=True
                    )

                else:

                    chart_df = results_df[
                        ["Model", "MAE", "RMSE", "R2 Score"]
                    ].copy()

                    chart_df = chart_df.set_index("Model")

                    st.bar_chart(
                        chart_df,
                        use_container_width=True
                    )

        # =====================================================
        # HYPERPARAMETER OPTIMIZATION
        # =====================================================

        st.header("⚙️ Hyperparameter Optimization")

        st.write(
            "Optimize selected models using cross-validation "
            "and randomized hyperparameter search."
        )

        if st.button("🔥 Optimize Models"):

            with st.spinner(
                "Running hyperparameter optimization..."
            ):

                if problem_type == "Classification":

                    models = get_classification_models()
                    param_grids = get_classification_param_grids()

                    # Keep only models that have tuning parameters
                    tuning_models = {
                        name: model
                        for name, model in models.items()
                        if name in param_grids
                    }

                    tuning_results = optimize_models(
                        X,
                        y,
                        preprocessor,
                        tuning_models,
                        param_grids,
                        scoring="f1_weighted"
                    )

                else:

                    models = get_regression_models()
                    param_grids = get_regression_param_grids()

                    tuning_models = {
                        name: model
                        for name, model in models.items()
                        if name in param_grids
                    }

                    tuning_results = optimize_models(
                        X,
                        y,
                        preprocessor,
                        tuning_models,
                        param_grids,
                        scoring="r2"
                    )

                tuning_df = pd.DataFrame(tuning_results)

                # Sort by best score
                tuning_df = tuning_df.sort_values(
                    by="Best Score",
                    ascending=False
                )

                st.subheader("🏆 Optimized Model Results")

                # Display only useful columns
                display_df = tuning_df[
                    [
                        "Model",
                        "Best Score",
                        "Best Parameters"
                    ]
                ]

                st.dataframe(
                    display_df,
                    use_container_width=True
                )
                # =====================================================
                # BEST MODEL SELECTION
                # =====================================================

                best_result = tuning_df.iloc[0]

                best_model_name = best_result["Model"]
                best_score = best_result["Best Score"]
                best_parameters = best_result["Best Parameters"]
                best_estimator = best_result["Best Estimator"]


                # =====================================================
                # RECOMMENDED MODEL
                # =====================================================

                best_result = tuning_df.iloc[0]

                best_model_name = best_result["Model"]
                best_score = best_result["Best Score"]
                best_parameters = best_result["Best Parameters"]
                best_estimator = best_result["Best Estimator"]


                # =====================================================
                # RECOMMENDED MODEL
                # =====================================================

                best_result = tuning_df.iloc[0]

                best_model_name = best_result["Model"]
                best_score = best_result["Best Score"]
                best_parameters = best_result["Best Parameters"]
                best_estimator = best_result["Best Estimator"]


                # =====================================================
                # RECOMMENDED MODEL
                # =====================================================

                st.header("🏆 Recommended Model")

                with st.container(border=True):

                    st.subheader(f"🏆 {best_model_name}")

                    st.write(
                        "Best performing model after hyperparameter optimization."
                    )

                    st.divider()

                    if problem_type == "Classification":

                        st.metric(
                            "Best CV F1 Score",
                            f"{best_score:.4f}"
                        )

                    else:

                        st.metric(
                            "Best CV R² Score",
                            f"{best_score:.4f}"
                        )

                    st.success("✨ Optimized & Recommended")


                # =====================================================
                # BEST HYPERPARAMETERS
                # =====================================================

                st.subheader("⚙️ Best Hyperparameters")

                with st.expander("View optimized parameters"):

                    st.json(best_parameters)


                # =====================================================
                # FINAL TEST PERFORMANCE
                # =====================================================

                st.header("📈 Final Test Performance")

                test_results = evaluate_best_model(
                    X,
                    y,
                    best_estimator,
                    problem_type
                )

                metric_cols = st.columns(len(test_results))

                for col, (metric, value) in zip(
                    metric_cols,
                    test_results.items()
                ):

                    with col:

                        st.metric(
                            metric,
                            f"{value:.4f}"
                        )
