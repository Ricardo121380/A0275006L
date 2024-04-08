import gurobipy as gp
from gurobipy import GRB
from itertools import product 
import numpy as np

# Parameters
num_trials = 100

num_customers = 15
num_facilities = 10

AFL_vals = []
AFL_LR_vals = []

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

  # AFL MIP model
  m_AFL = gp.Model("AFL")
  select = m_AFL.addVars(num_facilities, vtype=GRB.BINARY, name="Select")
  assign = m_AFL.addVars(cartesian_prod, ub=1, vtype=GRB.CONTINUOUS, name="Assign")

  m_AFL.addConstrs((gp.quicksum(assign[(c,f)] for c in range(num_customers)) <= num_customers*select[f] for f in range(num_facilities)), name="Setup2ship")
  m_AFL.addConstrs((gp.quicksum(assign[(c,f)] for f in range(num_facilities)) == 1 for c in range(num_customers)), name="Demand")

  m_AFL.setObjective(select.prod(setup_cost) + assign.prod(shipping_cost), GRB.MINIMIZE)

  m_AFL.optimize()
  AFL_vals.append(m_AFL.ObjVal)

  # AFL Linear Relaxation 
  m_AFL_LR = gp.Model("AFL_LR")
  select = m_AFL_LR.addVars(num_facilities, ub=1, name="Select")
  assign = m_AFL_LR.addVars(cartesian_prod, ub=1, name="Assign")

  m_AFL_LR.addConstrs((gp.quicksum(assign[(c,f)] for c in range(num_customers)) <= num_customers*select[f] for f in range(num_facilities)), name="Setup2ship")
  m_AFL_LR.addConstrs((gp.quicksum(assign[(c,f)] for f in range(num_facilities)) == 1 for c in range(num_customers)), name="Demand")

  m_AFL_LR.setObjective(select.prod(setup_cost) + assign.prod(shipping_cost), GRB.MINIMIZE)

  m_AFL_LR.optimize()
  AFL_LR_vals.append(m_AFL_LR.ObjVal)

print(len(AFL_vals), "AFL optimal values:", AFL_vals)
print(len(AFL_LR_vals), "AFL_LR optimal values:", AFL_LR_vals)  

num_tight = sum(v1 == v2 for v1, v2 in zip(AFL_vals, AFL_LR_vals))
print("\nAFL tight in", num_tight, "trials")