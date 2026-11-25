"""Exploratory plots and concise dataset observations."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .config import PLOT_DIR, REPORT_DIR, FEATURES
from .data_validation import load_data

def run_eda():
    df = load_data(); PLOT_DIR.mkdir(exist_ok=True); sns.set_theme(style="whitegrid")
    counts = df.Class.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6,4)); sns.barplot(x=["Legitimate", "Fraud"], y=[counts.get(0,0),counts.get(1,0)], ax=ax, palette=["#2583c4","#e04f5f"]); ax.set(title="Transactions by class", ylabel="Transactions", xlabel="Class"); fig.tight_layout(); fig.savefig(PLOT_DIR/"class_distribution.png",dpi=180); plt.close(fig)
    rate=100*counts.get(1,0)/len(df)
    fig, ax = plt.subplots(figsize=(5,3)); ax.bar(["Fraud"],[rate],color="#e04f5f"); ax.set(title="Fraud share of all transactions",ylabel="Percent of dataset",ylim=(0,max(1,rate*1.25))); ax.text(0,rate,f"{rate:.3f}%",ha="center",va="bottom"); fig.tight_layout(); fig.savefig(PLOT_DIR/"fraud_percentage.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4)); sns.histplot(data=df, x="Amount", hue="Class", bins=60, stat="density", common_norm=False, log_scale=(False,True), ax=ax); ax.set(title="Transaction amount distribution (log y)", xlabel="Amount"); fig.tight_layout(); fig.savefig(PLOT_DIR/"amount_distribution.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4)); sns.histplot(data=df, x="Time", hue="Class", bins=48, element="step", stat="count", common_norm=False, ax=ax); ax.set(title="Transactions over recorded time", xlabel="Seconds from dataset start"); fig.tight_layout(); fig.savefig(PLOT_DIR/"time_distribution.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,9)); sns.heatmap(df[FEATURES+['Class']].corr(), cmap="coolwarm", center=0, ax=ax); ax.set_title("Feature correlation"); fig.tight_layout(); fig.savefig(PLOT_DIR/"correlation_heatmap.png",dpi=180); plt.close(fig)
    selected=["V1","V4","V10","V14"]
    fig, axes=plt.subplots(2,2,figsize=(10,7));
    for c,ax in zip(selected,axes.flat): sns.boxplot(data=df,x="Class",y=c,showfliers=False,ax=ax); ax.set_title(f"{c} by class")
    fig.tight_layout(); fig.savefig(PLOT_DIR/"pca_feature_boxplots.png",dpi=180); plt.close(fig)
    obs={"fraud_percentage":float(100*counts.get(1,0)/len(df)),"fraud_amount_median":float(df.loc[df.Class==1,"Amount"].median()),"legitimate_amount_median":float(df.loc[df.Class==0,"Amount"].median()),"note":"V1–V28 are anonymised PCA components and have no directly interpretable business meaning. The dataset covers approximately two days and does not establish future-period generalisation."}
    (REPORT_DIR/"eda_observations.json").write_text(json.dumps(obs,indent=2),encoding="utf-8")
    return obs
