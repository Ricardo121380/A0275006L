import gurobipy as gp
from gurobipy import GRB
from itertools import product 
import numpy as np

# Parameters
num_trials = 100

num_customers = 15
num_facilities = 10

FLP_vals = []
FLP_LR_vals = []

for t in range(num_trials):

  # Generate random data
  customers = [(np.round(np.random.rand(),2), np.round(np.random.rand(),2)) for i in range(num_customers)]
  facilities = [(np.round(np.random.rand(),2), np.round(np.random.rand(),2)) for i in range(num_facilities)]

  setup_cost = [1.0 for i in range(num_facilities)]
  cost_per_mile = 1.0

  # Compute distances
  def compute_distance(loc1, loc2):
      dx = loc1[0] - loc2[0]
      dy = loc1[1] - loc2[1]
      return np.sqrt(dx**2 + dy**2)

  # Compute parameters
  cartesian_prod = list(product(range(num_customers), range(num_facilities)))
  shipping_cost = {(c,f) : cost_per_mile*compute_distance(customers[c], facilities[f]) for c, f in cartesian_prod}

  # FLP MIP model
  m_FLP = gp.Model("FLP")
  select = m_FLP.addVars(num_facilities, vtype=GRB.BINARY, name="Select")
  assign = m_FLP.addVars(cartesian_prod, ub=1, vtype=GRB.CONTINUOUS, name="Assign")

  m_FLP.addConstrs((assign[(c,f)] <= select[f] for c,f in cartesian_prod), name="Setup2ship")
  m_FLP.addConstrs((gp.quicksum(assign[(c,f)] for f in range(num_facilities)) == 1 for c in range(num_customers)), name="Demand")

  m_FLP.setObjective(select.prod(setup_cost) + assign.prod(shipping_cost), GRB.MINIMIZE)

  m_FLP.optimize()
  FLP_vals.append(m_FLP.ObjVal)

  # FLP Linear Relaxation 
  m_FLP_LR = gp.Model("FLP_LR")
  select = m_FLP_LR.addVars(num_facilities, ub=1, name="Select")
  assign = m_FLP_LR.addVars(cartesian_prod, ub=1, name="Assign")

  m_FLP_LR.addConstrs((assign[(c,f)] <= select[f] for c,f in cartesian_prod), name="Setup2ship")
  m_FLP_LR.addConstrs((gp.quicksum(assign[(c,f)] for f in range(num_facilities)) == 1 for c in range(num_customers)), name="Demand")

  m_FLP_LR.setObjective(select.prod(setup_cost) + assign.prod(shipping_cost), GRB.MINIMIZE)

  m_FLP_LR.optimize()
  FLP_LR_vals.append(m_FLP_LR.ObjVal)

print(len(FLP_vals), "FLP optimal values:", FLP_vals)
print(len(FLP_LR_vals), "FLP_LR optimal values:", FLP_LR_vals)  

num_tight = sum(v1 == v2 for v1, v2 in zip(FLP_vals, FLP_LR_vals))
print("\nFLP tight in", num_tight, "trials")