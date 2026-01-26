#!/usr/bin/env python3
"""Fast parallel trainer - maximizes GPU while maintaining safety."""
import os, sys, gc, json, pickle, time, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / 'models' / 'production'
FEATURES_DIR = PROJECT_ROOT / 'training_package' / 'features_cache'
MIN_ACC = 0.55

# Faster settings - reduced iterations but more aggressive
XGB = {'tree_method':'hist','device':'cuda','max_depth':8,'max_bin':128,'learning_rate':0.05,
       'subsample':0.8,'colsample_bytree':0.7,'objective':'binary:logistic','eval_metric':'auc','nthread':6}
LGB = {'device':'gpu','gpu_platform_id':0,'gpu_device_id':0,'max_depth':8,'num_leaves':127,
       'learning_rate':0.05,'feature_fraction':0.7,'bagging_fraction':0.8,'bagging_freq':5,
       'verbose':-1,'objective':'binary','metric':'auc','n_jobs':6}
CB = {'task_type':'GPU','devices':'0','depth':6,'iterations':300,'learning_rate':0.05,
      'early_stopping_rounds':20,'verbose':False,'loss_function':'Logloss','thread_count':6}

gpu_lock = threading.Lock()
progress = {'done': 0, 'total': 0, 'good': 0}

def needs_training(pair):
    rp = OUTPUT_DIR / f"{pair}_results.json"
    if not rp.exists(): return True
    try:
        with open(rp) as f: d = json.load(f)
        if d.get('_meta',{}).get('feature_count',0) == 0: return True
        acc = d.get('target_direction_10',{}).get('accuracy',0) or d.get('target_direction_5',{}).get('accuracy',0) or d.get('target_direction_1',{}).get('accuracy',0)
        return acc < MIN_ACC
    except: return True

def train(pair):
    import xgboost as xgb, lightgbm as lgb, catboost as cb
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    fp = FEATURES_DIR / f"{pair}_features.pkl"
    if not fp.exists():
        print(f"[SKIP] {pair}: No features")
        return None

    print(f"[TRAIN] {pair}: Loading...")
    with open(fp,'rb') as f: data = pickle.load(f)

    X_tr,X_v,X_te = data['X_train'],data['X_val'],data['X_test']
    y_tr,y_v,y_te = data['y_train'],data['y_val'],data['y_test']
    fn,tc = data['feature_names'],data['target_cols']

    print(f"[TRAIN] {pair}: {len(X_tr)} samples, {len(fn)} features")
    results,models = {},{}

    with gpu_lock:  # Serialize GPU access
        for t in tc[:3]:
            yt,yv,yte = y_tr[t],y_v[t],y_te[t]

            dt = xgb.DMatrix(X_tr,label=yt); dv = xgb.DMatrix(X_v,label=yv)
            xm = xgb.train(XGB,dt,400,[(dv,'v')],early_stopping_rounds=30,verbose_eval=False)

            td = lgb.Dataset(X_tr,label=yt); vd = lgb.Dataset(X_v,label=yv,reference=td)
            lm = lgb.train(LGB,td,400,[vd],callbacks=[lgb.early_stopping(30,verbose=False)])

            cm = cb.CatBoostClassifier(**CB); cm.fit(X_tr,yt,eval_set=(X_v,yv))

            dte = xgb.DMatrix(X_te)
            ep = (xm.predict(dte) + lm.predict(X_te) + cm.predict_proba(X_te)[:,1]) / 3
            ec = (ep > 0.5).astype(int)

            acc = accuracy_score(yte,ec)
            print(f"[TRAIN] {pair} {t}: {acc*100:.2f}%")
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

    acc10 = results.get('target_direction_10',{}).get('accuracy',0)
    progress['done'] += 1
    if acc10 >= MIN_ACC: progress['good'] += 1
    print(f"[DONE] {pair}: {acc10*100:.1f}% | Progress: {progress['done']}/{progress['total']} | Good: {progress['good']}")
    return results

def main():
    print("="*60)
    print("FAST PARALLEL TRAINER")
    print("="*60)

    # Find pairs needing training
    pairs = []
    for f in FEATURES_DIR.glob('*_features.pkl'):
        if 'mega' in f.name: continue
        pair = f.stem.replace('_features', '')
        if needs_training(pair):
            pairs.append(pair)

    progress['total'] = len(pairs)
    print(f"Pairs to train: {len(pairs)}")
    print()

    if not pairs:
        print("All done!")
        return

    start = time.time()

    # Train with thread pool (GPU lock ensures serialization)
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(train, pairs))

    elapsed = time.time() - start
    print()
    print("="*60)
    print(f"COMPLETE: {progress['done']} pairs in {elapsed/60:.1f} min")
    print(f"Good accuracy (>=55%): {progress['good']}")
    print("="*60)

if __name__ == '__main__':
    main()
