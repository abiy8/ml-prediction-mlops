"""Select on validation only, then evaluate the frozen winner on untouched test data."""
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train(output="artifacts/model.joblib", track=True):
    dataset=load_breast_cancer()
    # Original dataset labels benign=1. Explicitly use malignant=1 for interpretable risk probability.
    X=dataset.data; y=(dataset.target==0).astype(int)
    idx=np.arange(len(y))
    dev,test=train_test_split(idx,test_size=.2,stratify=y,random_state=42)
    train_idx,val=train_test_split(dev,test_size=.25,stratify=y[dev],random_state=42)
    candidates={"logistic_regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42)),
                "random_forest":RandomForestClassifier(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=1)}
    validation={}
    Path("data").mkdir(exist_ok=True)
    if track:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","sqlite:///data/mlflow.db"))
        client=mlflow.MlflowClient()
        if client.get_experiment_by_name("breast-cancer-baselines") is None:
            client.create_experiment("breast-cancer-baselines",artifact_location=Path("artifacts/mlflow").resolve().as_uri())
        mlflow.set_experiment("breast-cancer-baselines")
    for name,model in candidates.items():
        model.fit(X[train_idx],y[train_idx]);prob=model.predict_proba(X[val])[:,1]
        score=float(roc_auc_score(y[val],prob));validation[name]=score
        if track:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"seed":42,"model":name,"train_rows":len(train_idx),"validation_rows":len(val),"label":"malignant=1"})
                mlflow.log_metric("validation_roc_auc",score)
    winner=max(validation,key=validation.get);model=candidates[winner]
    # Retrain chosen configuration on train+validation before the single test evaluation.
    model.fit(X[dev],y[dev]);prob=model.predict_proba(X[test])[:,1];pred=(prob>=.5).astype(int)
    metrics={"roc_auc":float(roc_auc_score(y[test],prob)),"average_precision":float(average_precision_score(y[test],prob)),"accuracy":float(accuracy_score(y[test],pred)),"f1_malignant":float(f1_score(y[test],pred)),"confusion_matrix":confusion_matrix(y[test],pred).tolist()}
    report={"dataset":"Wisconsin Diagnostic Breast Cancer (scikit-learn bundled UCI data)","seed":42,"rows":len(y),"features":len(dataset.feature_names),"split":{"train":len(train_idx),"validation":len(val),"test":len(test)},"validation_roc_auc":validation,"selected_model":winner,"test":metrics,"positive_class":"malignant","usage":"Educational benchmark; not a medical diagnostic tool"}
    version=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()[:12]
    bundle={"model":model,"feature_names":list(dataset.feature_names),"reference":X[dev],"version":version,"report":report}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,path)
    Path("reports").mkdir(exist_ok=True);Path("reports/evaluation.json").write_text(json.dumps({**report,"model_version":version},indent=2))
    if track:
        with mlflow.start_run(run_name="selected-test-evaluation"):
            mlflow.log_params({"model":winner,"version":version,"test_rows":len(test)})
            mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v,float)})
            mlflow.log_artifact("reports/evaluation.json")
    return bundle

if __name__=="__main__":
    b=train(); print(json.dumps(b["report"],indent=2))
