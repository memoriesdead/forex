#!/usr/bin/env python3
"""Quick batch trainer for specific pairs - runs alongside main trainer."""
import os, sys, gc, json, pickle, time
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PAIRS = ['USDTRY', 'USDTHB', 'USDSGD', 'USDSEK', 'USDPLN', 'USDNOK', 'USDJPY', 'USDHUF', 'USDHKD', 'USDDKK',
         'USDCNH', 'USDCHF', 'USDCAD', 'SGDJPY', 'NZDUSD', 'NZDJPY', 'NZDCHF', 'NZDCAD']
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'production'
FEATURES_DIR = PROJECT_ROOT / 'training_package' / 'features_cache'

XGB = {'tree_method':'hist','device':'cuda','max_depth':10,'max_bin':256,'learning_rate':0.03,
       'subsample':0.8,'colsample_bytree':0.7,'objective':'binary:logistic','eval_metric':'auc','nthread':6}
LGB = {'device':'gpu','gpu_platform_id':0,'gpu_device_id':0,'max_depth':10,'num_leaves':255,
       'learning_rate':0.03,'feature_fraction':0.7,'bagging_fraction':0.8,'bagging_freq':5,
       'verbose':-1,'objective':'binary','metric':'auc','n_jobs':6}
CB = {'task_type':'GPU','devices':'0','depth':8,'iterations':500,'learning_rate':0.03,
      'early_stopping_rounds':30,'verbose':False,'loss_function':'Logloss','thread_count':6}

def train(pair):
    import xgboost as xgb, lightgbm as lgb, catboost as cb
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    fp = FEATURES_DIR / f"{pair}_features.pkl"
    if not fp.exists(): return None

    # Skip if already good
    rp = OUTPUT_DIR / f"{pair}_results.json"
    if rp.exists():
        with open(rp) as f: d = json.load(f)
        if d.get('_meta',{}).get('feature_count',0) > 0:
            acc = d.get('target_direction_10',{}).get('accuracy',0) or d.get('target_direction_5',{}).get('accuracy',0)
            if acc >= 0.55: return None

    print(f"[B2] {pair}: Loading...")
    with open(fp,'rb') as f: data = pickle.load(f)

    X_tr,X_v,X_te = data['X_train'],data['X_val'],data['X_test']
    y_tr,y_v,y_te = data['y_train'],data['y_val'],data['y_test']
    fn,tc = data['feature_names'],data['target_cols']

    print(f"[B2] {pair}: {len(X_tr)} samples, {len(fn)} features")
    results,models = {},{}

    for t in tc[:3]:
        yt,yv,yte = y_tr[t],y_v[t],y_te[t]

        dt = xgb.DMatrix(X_tr,label=yt); dv = xgb.DMatrix(X_v,label=yv)
        xm = xgb.train(XGB,dt,700,[(dv,'v')],early_stopping_rounds=40,verbose_eval=False)

        td = lgb.Dataset(X_tr,label=yt); vd = lgb.Dataset(X_v,label=yv,reference=td)
        lm = lgb.train(LGB,td,700,[vd],callbacks=[lgb.early_stopping(40,verbose=False)])

        cm = cb.CatBoostClassifier(**CB); cm.fit(X_tr,yt,eval_set=(X_v,yv))

        dte = xgb.DMatrix(X_te)
        ep = (xm.predict(dte) + lm.predict(X_te) + cm.predict_proba(X_te)[:,1]) / 3
        ec = (ep > 0.5).astype(int)

        acc = accuracy_score(yte,ec)
        print(f"[B2] {pair} {t}: {acc*100:.2f}%")
        results[t] = {'accuracy':float(acc),'auc':float(roc_auc_score(yte,ep)),'f1':float(f1_score(yte,ec))}
        models[t] = {'xgb':xm,'lgb':lm,'cb':cm}
        del dt,dv,dte

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass

    with open(OUTPUT_DIR/f"{pair}_models.pkl",'wb') as f: pickle.dump(models,f)
    results['_meta'] = {'feature_mode':'mega','feature_count':len(fn),'timestamp':datetime.now().isoformat()}
    with open(OUTPUT_DIR/f"{pair}_results.json",'w') as f: json.dump(results,f,indent=2)
    print(f"[B2] {pair}: DONE")
    return results

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '6'
    for p in PAIRS:
        try: train(p)
        except Exception as e: print(f"[B2] {p}: ERROR {e}")
