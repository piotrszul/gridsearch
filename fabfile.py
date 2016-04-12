import os
import gridsearch as gs
import sys
import glob
import yaml
from gridsearch import *
import xgboost as xgb

from sklearn.grid_search import ParameterGrid
from xbgcv import xgb_cv_task, xbg_defaults

def _merge_dict(d1,d2):
	d = d1.copy()
	d.update(d2)
	return d

def _complete(grid_f, default_f):
	def run():
		defaults = default_f()
		grid = grid_f()
		return [_merge_dict(defaults,param) for param in grid]
	return run

def _job_def():

	def get_data():
		data = pd.read_csv("data/train.csv")
		y_all = data['TARGET'].as_matrix()
		X_all = data.drop(['ID','TARGET'], axis=1).as_matrix()
		return xgb.DMatrix( X_all, label=y_all)

	def grid():
		return list(ParameterGrid({'max_depth':[3,4], 'eta':[0.05], 'subsample':[0.75],'colsample_bylevel':[1]}))

	return (_complete(grid,xbg_defaults) ,xgb_cv_task(get_data, num_round=20))


def test():
	os.system('pip list | grep numpy')
	print sys.path

def grid_show():
	grid,task = _job_def()
	for params in  grid():
		print params

def _execute(param, task_f):
		print "Running for params: %s" % param
		result = task_f(param)
		with open(os.path.join("tmp", "%s.yaml" % model_id(param)), 'w') as f:
			f.write(yaml.dump(result))
def run_seq():
	grid_f,task_f = _job_def()
	param_grid = grid_f()
	results = []
	print "Running sequential grid of %s params" %len(param_grid)
	for param in param_grid:
		_execute(param,task_f)

def run_task(index_str = None):
	index = int(index_str or "0")
	grid_f,task_f = _job_def()
	param_grid = grid_f()
	print "Running task %s grid of %s params" %(index, len(param_grid))
	param = param_grid[index]
	_execute(param,task_f)

def merge():
	def to_yaml(path):
		with open(path) as f:
			return yaml.load(f)
	df = gs.yaml_to_pandas([to_yaml(p) for p in glob.glob("tmp/*.yaml")])
	print df
	df.to_csv(os.path.join('tmp','result.csv'), index=False)	



