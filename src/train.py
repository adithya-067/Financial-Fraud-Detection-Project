"""Train, compare, tune the threshold, and persist an inference pipeline."""
import json, logging
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.inspection import permutation_importance
from .config import DATA_PATH, FEATURES, TARGET, RANDOM_STATE, MODEL_DIR, REPORT_DIR, PLOT_DIR
from .data_validation import load_data
from .preprocessing import make_preprocessor
from .threshold_tuning import select_threshold
from .evaluate import save_test_outputs, metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def train(path=DATA_PATH):
    df=load_data(path); X=df[FEATURES]; y=df[TARGET].astype(int)
    # Stratified random splits retain rare fraud cases in all partitions; this is not a future-time evaluation.
    X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=.20,random_state=RANDOM_STATE,stratify=y)
    X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,test_size=.25,random_state=RANDOM_STATE,stratify=y_dev)
    chronological=df.sort_values("Time",kind="stable")[TARGET]
    cut1=int(.60*len(chronological)); cut2=int(.80*len(chronological))
    chronology_check={name:{str(k):int(v) for k,v in part.value_counts().items()}
        for name,part in (("train",chronological.iloc[:cut1]),("validation",chronological.iloc[cut1:cut2]),("test",chronological.iloc[cut2:]))}
    models={"Dummy (most frequent)":DummyClassifier(strategy="most_frequent"),
            "Logistic Regression (balanced)":LogisticRegression(class_weight="balanced",max_iter=1500,C=1.0,random_state=RANDOM_STATE),
            "Logistic Regression (random undersampling)":LogisticRegression(max_iter=1500,C=1.0,random_state=RANDOM_STATE),
            "Random Forest (balanced)":RandomForestClassifier(n_estimators=180,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=RANDOM_STATE),
            "XGBoost (weighted)":None}
    try:
        from xgboost import XGBClassifier
        ratio=float((y_train==0).sum()/(y_train==1).sum())
        models["XGBoost (weighted)"]=XGBClassifier(n_estimators=220,max_depth=4,learning_rate=.06,subsample=.85,colsample_bytree=.85,
            reg_lambda=2,scale_pos_weight=ratio,eval_metric="logloss",n_jobs=-1,random_state=RANDOM_STATE)
    except ImportError: logging.warning("xgboost is unavailable; skipping that model")
    candidates=[]; rows=[]
    for name, estimator in models.items():
        if estimator is None: continue
        logging.info("Training %s",name)
        fit_X,fit_y=X_train,y_train
        if name.endswith("random undersampling)"):
            fraud_index=y_train[y_train==1].index
            legit_sample=X_train.loc[y_train==0].sample(n=len(fraud_index),random_state=RANDOM_STATE)
            fit_X=pd.concat([legit_sample,X_train.loc[fraud_index]]).sample(frac=1,random_state=RANDOM_STATE)
            fit_y=y_train.loc[fit_X.index]
        pipe=Pipeline([("preprocess",make_preprocessor()),("model",estimator)])
        if name=="Logistic Regression (balanced)":
            # Small, deterministic CV search on training data only; refit the chosen C on all training rows.
            search_X,_,search_y,_=train_test_split(X_train,y_train,train_size=min(50000,len(X_train)),
                random_state=RANDOM_STATE,stratify=y_train)
            search=RandomizedSearchCV(pipe,{"model__C":[0.1,1.0,10.0]},n_iter=3,scoring="average_precision",
                cv=2,random_state=RANDOM_STATE,n_jobs=1,refit=False)
            search.fit(search_X,search_y)
            pipe.set_params(model__C=float(search.best_params_["model__C"]))
        pipe.fit(fit_X,fit_y)
        prob=pipe.predict_proba(X_val)[:,1]; pred=prob>=.5
        tn,fp,fn,tp=confusion_matrix(y_val,pred,labels=[0,1]).ravel()
        row={"model":name,"precision":precision_score(y_val,pred,zero_division=0),"recall":recall_score(y_val,pred,zero_division=0),
             "f1":f1_score(y_val,pred,zero_division=0),"roc_auc":roc_auc_score(y_val,prob),"pr_auc":average_precision_score(y_val,prob),
             "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
        rows.append(row); candidates.append((name,pipe,row))
    # PR-AUC ranks candidate probability quality; threshold and final metrics use validation and untouched test partitions respectively.
    comparison=pd.DataFrame(rows).sort_values("pr_auc",ascending=False)
    comparison.to_csv(REPORT_DIR/"model_comparison.csv",index=False)
    chosen_name=str(comparison.iloc[0].model); chosen=next(x[1] for x in candidates if x[0]==chosen_name)
    val_prob=chosen.predict_proba(X_val)[:,1]; threshold,table=select_threshold(y_val.to_numpy(),val_prob,min_recall=.80)
    table.to_csv(REPORT_DIR/"threshold_sweep.csv",index=False)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(table.threshold,table.precision,label="Precision"); ax.plot(table.threshold,table.recall,label="Recall"); ax.plot(table.threshold,table.f1,label="F1"); ax.axvline(threshold,color="black",linestyle="--",label=f"Selected {threshold:.3f}"); ax.set(xlabel="Decision threshold",ylabel="Score",title="Validation threshold trade-off"); ax.legend(); fig.tight_layout(); fig.savefig(PLOT_DIR/"threshold_tradeoff.png",dpi=170); plt.close(fig)
    # Permutation importance on a bounded validation sample describes model reliance, not causation.
    explain_n=min(5000,len(X_val)); explain_X=X_val.iloc[:explain_n]; explain_y=y_val.iloc[:explain_n]
    importance=permutation_importance(chosen,explain_X,explain_y,scoring="average_precision",n_repeats=3,
        random_state=RANDOM_STATE,n_jobs=1)
    importances=pd.DataFrame({"feature":FEATURES,"importance_mean":importance.importances_mean,
        "importance_std":importance.importances_std}).sort_values("importance_mean",ascending=False)
    importances.to_csv(REPORT_DIR/"permutation_importance.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,7)); top=importances.head(15).sort_values("importance_mean")
    ax.barh(top.feature,top.importance_mean,xerr=top.importance_std,color="#3aa7c6"); ax.set(xlabel="Decrease in validation PR-AUC after permutation",title="Permutation importance (validation sample)"); fig.tight_layout(); fig.savefig(PLOT_DIR/"permutation_importance.png",dpi=170); plt.close(fig)
    test_prob=chosen.predict_proba(X_test)[:,1]
    test_metrics=save_test_outputs(y_test.to_numpy(),test_prob,threshold,REPORT_DIR)
    metadata={"model_name":chosen_name,"threshold":threshold,"features":FEATURES,"dataset_rows":int(len(df)),
        "demo_samples":{"legitimate":{k:float(v) for k,v in df.loc[df[TARGET]==0,FEATURES].iloc[0].items()},
                        "fraud":{k:float(v) for k,v in df.loc[df[TARGET]==1,FEATURES].iloc[0].items()}},
        "class_counts":{str(k):int(v) for k,v in y.value_counts().items()},"fraud_percentage":float(100*y.mean()),
        "split":{"method":"stratified random","train":int(len(X_train)),"validation":int(len(X_val)),"test":int(len(X_test)),"random_state":RANDOM_STATE},
        "chronological_split_class_counts_diagnostic":chronology_check,
        "selection":"Highest validation PR-AUC at default threshold; threshold selected on validation to maximize precision subject to recall >= 0.80. Test set used once for final reporting.",
        "training_configuration":{"seed":RANDOM_STATE,"test_fraction":0.20,"validation_fraction_of_development":0.25,
            "amount_transform":"log1p then standard scaling","other_features":"standard scaling",
            "logistic_search":"3 C values, 2-fold average-precision CV on at most 50000 training rows"},
        "validation_comparison":rows,"validation_threshold_metrics":metrics(y_val.to_numpy(),val_prob,threshold),"test_metrics":test_metrics}
    joblib.dump({"pipeline":chosen,"threshold":threshold,"metadata":metadata},MODEL_DIR/"fraud_pipeline.joblib")
    (MODEL_DIR/"model_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (REPORT_DIR/"training_summary.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    logging.info("Saved model %s; test PR-AUC %.4f",chosen_name,test_metrics["pr_auc"])
    return metadata

if __name__=="__main__": train()
