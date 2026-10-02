import json
import sqlite3
import uuid
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, roc_auc_score

class Monitor:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.path=path
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, features TEXT, probability REAL, version TEXT, label INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    def connect(self):
        c=sqlite3.connect(self.path,timeout=10);c.execute("PRAGMA journal_mode=WAL");return c
    def record(self,rows,prob,version):
        ids=[str(uuid.uuid4()) for _ in rows]
        with self.connect() as c:
            c.executemany("INSERT INTO predictions(id,features,probability,version) VALUES (?,?,?,?)",[(i,json.dumps(x),float(p),version) for i,x,p in zip(ids,rows,prob)])
        return ids
    def feedback(self,id,label):
        with self.connect() as c:
            cursor=c.execute("UPDATE predictions SET label=? WHERE id=?",(label,id))
            return cursor.rowcount==1
    def report(self,reference,names,version,limit=1000):
        with self.connect() as c:
            rows=c.execute("SELECT features,probability,label FROM predictions WHERE version=? ORDER BY rowid DESC LIMIT ?",(version,limit)).fetchall()
        labeled=[r for r in rows if r[2] is not None]
        quality={"labeled_count":len(labeled)}
        if labeled:
            y=[r[2] for r in labeled];p=[r[1] for r in labeled]
            quality["accuracy"]=float(accuracy_score(y,np.array(p)>=.5))
            quality["roc_auc"]=float(roc_auc_score(y,p)) if len(set(y))==2 else None
        drift=[]
        if len(rows)>=30:
            recent=np.array([json.loads(r[0]) for r in rows]);alpha=.05/len(names)
            for i,name in enumerate(names):
                result=ks_2samp(reference[:,i],recent[:,i])
                drift.append({"feature":name,"ks_statistic":float(result.statistic),"p_value":float(result.pvalue),"flagged":bool(result.pvalue<alpha and result.statistic>.2)})
        return {"model_version":version,"window_count":len(rows),"minimum_drift_samples":30,"drift":drift,"quality":quality,"drift_rule":"KS statistic > 0.2 and Bonferroni-adjusted p < 0.05; heuristic alert, not proof of performance degradation"}
