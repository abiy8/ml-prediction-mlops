import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
import joblib
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from .monitor import Monitor

def authorize(x_api_key: str | None = Header(default=None)):
    key=os.getenv("API_KEY","")
    if key:
        import hmac
        if not x_api_key or not hmac.compare_digest(key,x_api_key): raise HTTPException(401,"Invalid API key")

class Prediction(BaseModel):
    rows: list[list[float]]=Field(min_length=1,max_length=100)
    @field_validator("rows")
    @classmethod
    def finite(cls,rows):
        if any(not np.isfinite(v) for row in rows for v in row):raise ValueError("Features must be finite")
        return rows
class Feedback(BaseModel):
    prediction_id: str
    label: int=Field(ge=0,le=1)

def create_app(model_path=None,db_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path=Path(model_path or os.getenv("MODEL_PATH","artifacts/model.joblib"))
        # Load only trusted locally trained artifacts. Joblib can execute code.
        app.state.bundle=joblib.load(path) if path.exists() else None
        app.state.monitor=Monitor(db_path or os.getenv("MONITOR_DB","data/monitor.db"))
        yield
    app=FastAPI(title="ML Prediction and Monitoring",lifespan=lifespan)
    def bundle():
        if app.state.bundle is None:raise HTTPException(503,"Train a model with python train.py before serving predictions")
        return app.state.bundle
    @app.get("/")
    def index():return FileResponse(Path(__file__).parent/"web.html")
    @app.get("/health")
    def health():return {"ready":app.state.bundle is not None,"model_version":app.state.bundle["version"] if app.state.bundle else None}
    @app.get("/model",dependencies=[Depends(authorize)])
    def model():
        b=bundle();return {"version":b["version"],"feature_names":b["feature_names"],"report":b["report"]}
    @app.post("/predict",dependencies=[Depends(authorize)])
    def predict(q:Prediction):
        b=bundle()
        if any(len(row)!=len(b["feature_names"]) for row in q.rows):raise HTTPException(422,f"Expected {len(b['feature_names'])} features per row in /model order")
        t=time.perf_counter();prob=b["model"].predict_proba(q.rows)[:,1];ids=app.state.monitor.record(q.rows,prob,b["version"])
        return {"model_version":b["version"],"latency_ms":(time.perf_counter()-t)*1000,"predictions":[{"id":i,"malignant_probability":float(p),"label":int(p>=.5)} for i,p in zip(ids,prob)]}
    @app.post("/feedback",dependencies=[Depends(authorize)])
    def feedback(q:Feedback):
        if not app.state.monitor.feedback(q.prediction_id,q.label):raise HTTPException(404,"Prediction not found")
        return {"recorded":True}
    @app.get("/monitor",dependencies=[Depends(authorize)])
    def monitor():
        b=bundle();return app.state.monitor.report(b["reference"],b["feature_names"],b["version"])
    return app
app=create_app()
