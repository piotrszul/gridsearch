import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import xgboost as xgb
from sklearn.grid_search import ParameterGrid
import yaml
import itertools

DEF_BST_PARAMS = {'eta':0.3, 
              'gamma': 0,
              'max_depth': 6,
              'min_child_weight': 1,
              'max_delta_step': 0,
              'subsample': 1.0,
              'colsample_bytree': 1.0,
              'colsample_bylevel': 1.0,
              'lambda': 1.0,
              'alpha': 0.0,
              'sketch_eps': 0.03
             }

def model_id(bst_params):
    return "+".join(sorted([ str(k) + "=" + str(v) for k,v in bst_params.items()]))

def xgb_cv(data, bst_params={}, num_round=10, nfold=3):
    this_params = {'silent':1, 'objective':'binary:logistic', 'seed':10, 'eval_metric':'auc', 'nthread':4 }
    this_bst_params = DEF_BST_PARAMS.copy()
    this_bst_params.update(bst_params)
    this_params.update(dict(map(lambda (k,v): ('bst:' + k, v), this_bst_params.items())))
    print this_params
    print this_bst_params
    cv_model = xgb.cv(this_params, data, num_round, 
                      nfold=nfold,metrics=['auc'], seed = 0, show_progress=10, early_stopping_rounds=20)
    best_iter = np.argmax(cv_model['test-auc-mean'])
    print best_iter
    return cv_model, best_iter, this_bst_params
    
def run_task(tast_index):
    task_params =  grid_params[tast_index]
    cv_model, best_iter, this_bst_params = xgb_cv(dall,task_params)
    cv_model[best_iter:best_iter+1]
    best_aux = cv_model[best_iter:best_iter+1]
    result = {'bst_params':this_bst_params,
          'best_iteration':best_iter, 
          'aux':dict(map(lambda (k,v): (k, float(v[best_iter])), best_aux.to_dict().items()))}
    return result


def flatten(l):
    return list(itertools.chain.from_iterable(l))

def flatten_dict(d):
    def agg_key(key, subdict):
        if type(subdict) is dict:
            return [ (key + "." + k, v) for (k,v) in subdict.iteritems()]
        else:
            return [(key, subdict)]
    if type(d) is dict:
        return dict(flatten([ agg_key(k,flatten_dict(v)) for (k,v) in d.iteritems()]))
    else:
        return d
    
def yaml_to_pandas(seq):
    #map all the keys first
    result = []
    for r in seq:
        row  = flatten_dict(r)
 #       print row
        result.append(row)
    return pd.DataFrame(result)

def hello():
	print "Hello"