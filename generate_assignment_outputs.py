"""Run the project notebook and create the curated report artifacts.

The notebook remains the main analysis source. This helper executes its code
cells, then saves only the tables and figures needed for Parts C-G of the
assignment.
"""

import json
import os
import shutil
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Phishing_Website_Classification_and_Clustering.ipynb"
OUTPUT_DIR = ROOT / "outputs_phishing"


def execute_notebook():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"Running notebook cell {index}...", flush=True)
        try:
            exec(compile(source, f"notebook_cell_{index}", "exec"), namespace)
        except Exception:
            print(f"Notebook cell {index} failed.", flush=True)
            traceback.print_exc()
            raise
    return namespace


def clean_outputs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for item in list(OUTPUT_DIR.iterdir()):
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def curate(namespace):
    raw = namespace["raw"]
    df = namespace["df"]
    X_all = namespace["X_all"]
    y_all = namespace["y_all"]
    X_test = namespace["X_test"]
    y_test = namespace["y_test"]
    models = namespace["models"]
    baseline_results = namespace["baseline_results"]
    final_results = namespace["final_results"]
    cluster_scores = namespace["cluster_scores"]
    profile = namespace["profile"]
    cluster_feature_profile = namespace["cluster_feature_profile"]
    proj = namespace["proj"]
    cluster_ids = namespace["cluster_ids"]
    best_k = namespace["best_k"]
    cv_scores = namespace["cv_scores"]
    search = namespace["search"]
    importance = namespace["importance"]
    RANDOM_STATE = namespace["RANDOM_STATE"]
    rf_pipeline = namespace["rf_pipeline"]
    evaluate_model = namespace["evaluate_model"]
    roc_auc_score = namespace["roc_auc_score"]
    roc_curve = namespace["roc_curve"]
    ConfusionMatrixDisplay = namespace["ConfusionMatrixDisplay"]

    # Part C: dataset summary and feature provenance audit.
    dataset_summary = pd.DataFrame({
        "Item": [
            "Original rows", "Rows after duplicate-URL removal", "Original columns",
            "Usable numeric features", "Legitimate websites", "Phishing websites",
            "Missing values", "Duplicate full rows", "Repeated URL rows before removal",
        ],
        "Value": [
            len(raw), len(df), raw.shape[1], X_all.shape[1],
            int((y_all == 0).sum()), int((y_all == 1).sum()),
            int(raw.isna().sum().sum()), int(raw.duplicated().sum()),
            int(raw.duplicated(subset="URL").sum()),
        ],
    })
    dataset_summary.to_csv(OUTPUT_DIR / "01_dataset_summary.csv", index=False)

    provenance = {
        "URLSimilarityIndex": "Precomputed similarity-based feature",
        "URLCharProb": "Derived probability feature",
        "TLDLegitimateProb": "Derived legitimacy-probability feature",
        "LineOfCode": "Webpage-derived feature",
        "LargestLineLength": "Webpage-derived feature",
        "NoOfImage": "Webpage-derived feature",
        "NoOfCSS": "Webpage-derived feature",
        "NoOfJS": "Webpage-derived feature",
        "NoOfSelfRef": "Webpage-derived feature",
        "NoOfExternalRef": "Webpage-derived feature",
    }
    audit_rows = []
    for feature in X_all.columns:
        values = X_all[feature].astype(float)
        auc = roc_auc_score(y_all, values)
        auc = max(float(auc), 1.0 - float(auc))
        audit_rows.append({
            "Feature": feature,
            "Absolute_target_correlation": abs(float(values.corr(y_all))),
            "Univariate_ROC_AUC": auc,
            "Unique_values": int(values.nunique()),
            "Feature_provenance": provenance.get(feature, "URL or numeric website feature"),
        })
    feature_audit = pd.DataFrame(audit_rows).sort_values("Univariate_ROC_AUC", ascending=False)
    feature_audit.to_csv(OUTPUT_DIR / "02_feature_audit.csv", index=False)

    usi = df["URLSimilarityIndex"]
    similarity_audit = pd.DataFrame([
        {
            "Rule": "URLSimilarityIndex < 100",
            "Rows": int((usi < 100).sum()),
            "Phishing_rate": float(y_all[usi < 100].mean()),
        },
        {
            "Rule": "URLSimilarityIndex == 100",
            "Rows": int((usi == 100).sum()),
            "Phishing_rate": float(y_all[usi == 100].mean()),
        },
        {
            "Rule": "80 <= URLSimilarityIndex < 100",
            "Rows": int(((usi >= 80) & (usi < 100)).sum()),
            "Phishing_rate": float(y_all[(usi >= 80) & (usi < 100)].mean()),
        },
    ])
    similarity_audit.to_csv(OUTPUT_DIR / "03_similarity_feature_audit.csv", index=False)

    # Part C: EDA figures.
    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts = y_all.map({0: "Legitimate", 1: "Phishing"}).value_counts().reindex(["Legitimate", "Phishing"])
    class_counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("Class distribution")
    ax.set_xlabel("Website class")
    ax.set_ylabel("Number of websites")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_eda_class_distribution.png", dpi=180)
    plt.close(fig)

    eda_features = [c for c in ["URLLength", "DomainLength", "NoOfSubDomain", "LineOfCode"] if c in X_all.columns]
    eda_sample = X_all[eda_features].sample(n=min(10000, len(X_all)), random_state=RANDOM_STATE).copy()
    eda_sample["Class"] = y_all.loc[eda_sample.index].map({0: "Legitimate", 1: "Phishing"})
    fig, axes = plt.subplots(1, len(eda_features), figsize=(4.2 * len(eda_features), 3.8))
    axes = np.atleast_1d(axes)
    for ax, feature in zip(axes, eda_features):
        eda_sample.boxplot(column=feature, by="Class", ax=ax, showfliers=False)
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Value")
    fig.suptitle("Selected feature distributions by class")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_eda_feature_distributions.png", dpi=180)
    plt.close(fig)

    top_corr = X_all.corrwith(y_all).abs().sort_values(ascending=False).dropna().head(12).index.tolist()
    corr_matrix = pd.concat([X_all[top_corr], y_all.rename("phishing_target")], axis=1).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr_matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=85, fontsize=8)
    ax.set_yticks(np.arange(len(corr_matrix.index)), corr_matrix.index, fontsize=8)
    fig.colorbar(image, ax=ax, shrink=.7, label="Correlation")
    ax.set_title("Correlation matrix of strongest features")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_eda_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    # Part D: baseline supervised results and figures.
    baseline_results.reset_index().to_csv(OUTPUT_DIR / "07_baseline_model_results.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, ["Logistic Regression (baseline)", "Random Forest (baseline)"]):
        ConfusionMatrixDisplay.from_predictions(
            y_test, models[name].predict(X_test), labels=[0, 1],
            display_labels=["Legitimate", "Phishing"], ax=ax, colorbar=False,
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_supervised_confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ["Logistic Regression (baseline)", "Random Forest (baseline)"]:
        probability = models[name].predict_proba(X_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)
        ax.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={roc_auc_score(y_test, probability):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves for baseline models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_supervised_roc_curves.png", dpi=180)
    plt.close(fig)

    # Part E: clustering results and figures.
    cluster_scores.to_csv(OUTPUT_DIR / "10_cluster_selection_scores.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(cluster_scores["K"], cluster_scores["Inertia"], "o-")
    axes[0].set_title("Elbow method")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(cluster_scores["K"], cluster_scores["Silhouette"], "o-")
    axes[1].set_title("Silhouette scores")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette score")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_cluster_selection.png", dpi=180)
    plt.close(fig)
    profile.reset_index().to_csv(OUTPUT_DIR / "12_cluster_profiles.csv", index=False)
    cluster_feature_profile.reset_index().to_csv(OUTPUT_DIR / "13_cluster_feature_profiles.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=cluster_ids, s=8, alpha=.55, cmap="tab10")
    ax.set_title(f"K-Means feature clusters (K={best_k})")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "14_cluster_pca_visualization.png", dpi=180)
    plt.close(fig)

    # Part F: cross-validation, tuning, feature selection, and a proxy-feature ablation.
    cv_table = pd.DataFrame({"Fold": np.arange(1, len(cv_scores) + 1), "F1_phishing": cv_scores})
    cv_table.loc[len(cv_table)] = ["Mean", cv_scores.mean()]
    cv_table.loc[len(cv_table)] = ["Std", cv_scores.std()]
    cv_table.to_csv(OUTPUT_DIR / "15_cross_validation_results.csv", index=False)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(OUTPUT_DIR / "16_hyperparameter_search_results.csv", index=False)
    importance.reset_index(name="Importance").rename(columns={"index": "Feature"}).to_csv(OUTPUT_DIR / "17_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.head(15).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Random Forest top-15 feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "18_feature_importance.png", dpi=180)
    plt.close(fig)

    proxy_features = [
        "URLSimilarityIndex", "URLCharProb", "TLDLegitimateProb",
        "URLTitleMatchScore", "DomainTitleMatchScore",
    ]
    reduced_features = [c for c in X_all.columns if c not in proxy_features]
    reduced_model = rf_pipeline()
    reduced_model.fit(namespace["X_train"][reduced_features], namespace["y_train"])
    reduced_result = evaluate_model(
        "Random Forest (without proxy features)", reduced_model,
        X_test[reduced_features], y_test,
    )[0]
    reduced_results = pd.DataFrame([reduced_result])
    reduced_results.to_csv(OUTPUT_DIR / "19_proxy_feature_ablation.csv", index=False)
    reduced_metrics = {
        key: reduced_result[key]
        for key in [
            "Accuracy", "Precision_phishing", "Recall_phishing", "F1_phishing",
            "ROC_AUC_phishing", "Specificity_legitimate", "False_positive", "False_negative",
        ]
    }

    def metric_row(name):
        return final_results.loc[name, [
            "Accuracy", "Precision_phishing", "Recall_phishing",
            "F1_phishing", "ROC_AUC_phishing", "Specificity_legitimate",
            "False_positive", "False_negative",
        ]].to_dict()

    improvement_table = pd.DataFrame([
        {"Model_version": "RF baseline", "Technique": "Baseline", "Features": "50 numeric features", **metric_row("Random Forest (baseline)")},
        {"Model_version": "RF tuned", "Technique": "RandomizedSearchCV", "Features": "50 numeric features", **metric_row("Random Forest (tuned)")},
        {"Model_version": "RF top 15", "Technique": "Feature selection", "Features": "Top 15 RF features", **metric_row("Random Forest (top 15 features)")},
        {"Model_version": "RF without proxy features", "Technique": "Feature engineering/ablation", "Features": f"{len(reduced_features)} reduced features", **reduced_metrics},
    ])
    improvement_table.to_csv(OUTPUT_DIR / "20_improvement_comparison.csv", index=False)
    error_table = improvement_table[["Model_version", "False_positive", "False_negative"]].copy()
    error_table.to_csv(OUTPUT_DIR / "21_improvement_error_counts.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    improvement_table.set_index("Model_version")[["Accuracy", "F1_phishing"]].plot(
        kind="bar", ax=axes[0], ylim=(0.999, 1.0001), rot=20,
    )
    axes[0].set_title("Scores near the benchmark ceiling")
    axes[0].set_ylabel("Score")
    error_table.set_index("Model_version")[["False_positive", "False_negative"]].plot(
        kind="bar", ax=axes[1], rot=20,
    )
    axes[1].set_title("Held-out test errors")
    axes[1].set_ylabel("Number of errors")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "21_improvement_comparison.png", dpi=180)
    plt.close(fig)

    # Part G: paper comparison. Use baseline models for the closest comparison.
    paper_comparison = pd.DataFrame([
        {
            "Model": "Random Forest",
            "Paper_Accuracy": .9333, "Our_Accuracy": baseline_results.loc["Random Forest (baseline)", "Accuracy"],
            "Paper_Precision": .9231, "Our_Precision": baseline_results.loc["Random Forest (baseline)", "Precision_phishing"],
            "Paper_Recall": .9524, "Our_Recall": baseline_results.loc["Random Forest (baseline)", "Recall_phishing"],
            "Paper_F1": .9375, "Our_F1": baseline_results.loc["Random Forest (baseline)", "F1_phishing"],
            "Paper_ROC_AUC": .9238, "Our_ROC_AUC": baseline_results.loc["Random Forest (baseline)", "ROC_AUC_phishing"],
        },
        {
            "Model": "Logistic Regression",
            "Paper_Accuracy": .6583, "Our_Accuracy": baseline_results.loc["Logistic Regression (baseline)", "Accuracy"],
            "Paper_Precision": .8667, "Our_Precision": baseline_results.loc["Logistic Regression (baseline)", "Precision_phishing"],
            "Paper_Recall": .6118, "Our_Recall": baseline_results.loc["Logistic Regression (baseline)", "Recall_phishing"],
            "Paper_F1": .7172, "Our_F1": baseline_results.loc["Logistic Regression (baseline)", "F1_phishing"],
            "Paper_ROC_AUC": .6610, "Our_ROC_AUC": baseline_results.loc["Logistic Regression (baseline)", "ROC_AUC_phishing"],
        },
    ])
    paper_comparison.to_csv(OUTPUT_DIR / "22_paper_comparison.csv", index=False)

    artifact_guide = pd.DataFrame([
        {"Artifact": "01_dataset_summary.csv", "Report_section": "4.1 Dataset acquisition", "Use": "Dataset size, labels, and cleaning counts"},
        {"Artifact": "02_feature_audit.csv", "Report_section": "3.5 Feature-engineering limitations", "Use": "Feature strength and provenance"},
        {"Artifact": "03_similarity_feature_audit.csv", "Report_section": "3.5 Feature-engineering limitations / 11 Discussion", "Use": "Evidence for the similarity-feature concern"},
        {"Artifact": "04_eda_class_distribution.png", "Report_section": "5.2 Class distribution", "Use": "Class-count bar chart"},
        {"Artifact": "05_eda_feature_distributions.png", "Report_section": "5.4 Feature distributions", "Use": "Selected feature boxplots"},
        {"Artifact": "06_eda_correlation_heatmap.png", "Report_section": "5.5 Correlation analysis", "Use": "Correlation visualization"},
        {"Artifact": "07_baseline_model_results.csv", "Report_section": "7 Supervised results", "Use": "Accuracy, precision, recall, F1, and ROC-AUC"},
        {"Artifact": "08_supervised_confusion_matrices.png", "Report_section": "7 Supervised results", "Use": "False positives and false negatives"},
        {"Artifact": "09_supervised_roc_curves.png", "Report_section": "7 Supervised results", "Use": "ROC curves and AUC"},
        {"Artifact": "10_cluster_selection_scores.csv / 11_cluster_selection.png", "Report_section": "8.2 Choosing K", "Use": "Elbow and silhouette evidence"},
        {"Artifact": "12_cluster_profiles.csv / 13_cluster_feature_profiles.csv", "Report_section": "8.4 Cluster interpretation", "Use": "Cluster size and feature profiles"},
        {"Artifact": "14_cluster_pca_visualization.png", "Report_section": "8.3 Cluster visualization", "Use": "PCA cluster plot"},
        {"Artifact": "15_cross_validation_results.csv", "Report_section": "9.1 Cross-validation", "Use": "Fold scores, mean, and standard deviation"},
        {"Artifact": "16_hyperparameter_search_results.csv", "Report_section": "9.2 Hyperparameter tuning", "Use": "Search configurations and scores"},
        {"Artifact": "17_feature_importance.csv / 18_feature_importance.png", "Report_section": "9.3 Feature selection", "Use": "Top-feature ranking"},
        {"Artifact": "19_proxy_feature_ablation.csv", "Report_section": "9.3 Feature selection / 11 Discussion", "Use": "Full versus reduced feature set"},
        {"Artifact": "20_improvement_comparison.csv / 21_improvement_comparison.png", "Report_section": "9.4 Improvement comparison", "Use": "Baseline versus improved models and held-out errors"},
        {"Artifact": "21_improvement_error_counts.csv", "Report_section": "9.4 Improvement comparison", "Use": "Exact false-positive and false-negative counts"},
        {"Artifact": "22_paper_comparison.csv", "Report_section": "10 Comparison with published paper", "Use": "Paper versus reproduced metrics"},
    ])
    artifact_guide.to_csv(OUTPUT_DIR / "23_artifact_guide.csv", index=False)
    (OUTPUT_DIR / "README.md").write_text(
        "These are the curated artifacts for the assignment report. Use 23_artifact_guide.csv to map each file to its report section.\n",
        encoding="utf-8",
    )

    print("Curated artifacts written to:", OUTPUT_DIR)
    print("Selected clustering K:", best_k)
    print("Proxy-feature ablation:")
    print(reduced_results.round(4).to_string(index=False))


if __name__ == "__main__":
    namespace = execute_notebook()
    # The final notebook cell performs curation when the notebook is run
    # normally. Keep this fallback for older copies of the notebook.
    if not (OUTPUT_DIR / "23_artifact_guide.csv").exists():
        clean_outputs()
        curate(namespace)
