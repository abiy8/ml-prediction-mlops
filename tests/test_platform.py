import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer
from train import train
from app.main import create_app
from app.monitor import Monitor

@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    p=tmp_path_factory.mktemp("model")/"model.joblib";b=train(str(p),track=False);return p,b

def test_evaluation_is_reproducible_and_split_balanced(bundle,tmp_path):
    _,b=bundle;other=train(str(tmp_path/"b.joblib"),track=False)
    assert b["report"]==other["report"]
    assert sum(b["report"]["split"].values())==569
    assert b["report"]["test"]["roc_auc"]>.95

def test_api_schema_feedback_and_auth(bundle,tmp_path,monkeypatch):
    monkeypatch.setenv("API_KEY","test")
    with TestClient(create_app(str(bundle[0]),str(tmp_path/"m.db"))) as c:
        headers={"X-API-Key":"test"}
        assert c.get("/model").status_code==401
        assert c.post("/predict",headers=headers,json={"rows":[[1,2]]}).status_code==422
        r=c.post("/predict",headers=headers,json={"rows":[load_breast_cancer().data[0].tolist()]}).json()
        assert 0<=r["predictions"][0]["malignant_probability"]<=1
        assert c.post("/feedback",headers=headers,json={"prediction_id":r["predictions"][0]["id"],"label":1}).status_code==200
        assert c.get("/monitor",headers=headers).json()["quality"]["labeled_count"]==1
        assert c.post("/feedback",headers=headers,json={"prediction_id":"unknown","label":0}).status_code==404

def test_drift_detects_large_shift_and_isolates_versions(tmp_path):
    rng=np.random.default_rng(42);ref=rng.normal(size=(400,2));m=Monitor(str(tmp_path/"m.db"))
    m.record((ref[:40]+5).tolist(),[.5]*40,"v1")
    assert all(x["flagged"] for x in m.report(ref,["x","y"],"v1")["drift"])
    assert m.report(ref,["x","y"],"v2")["window_count"]==0

def test_untrained_service_reports_unready(tmp_path):
    with TestClient(create_app(str(tmp_path/"absent"),str(tmp_path/"m.db"))) as c:
        assert not c.get("/health").json()["ready"]
        assert c.post("/predict",json={"rows":[[1]]}).status_code==503


def test_mlflow_records_candidates_and_test_run(tmp_path,monkeypatch):
    import mlflow
    monkeypatch.setenv('MLFLOW_TRACKING_URI','sqlite:///'+str(tmp_path/'tracking.db'))
    train(str(tmp_path/'tracked.joblib'),track=True)
    runs=mlflow.search_runs(experiment_names=['breast-cancer-baselines'])
    assert len(runs)==3
    assert all('artifacts/mlflow' in uri for uri in runs['artifact_uri'])
    assert runs['metrics.validation_roc_auc'].notna().sum()==2
    assert runs['metrics.roc_auc'].notna().sum()==1
