import gurobipy as gp
from gurobipy import GRB
from itertools import product 
import numpy as np

# Parameters
num_trials = 100
num_customers = 15
num_facilities = 10
capacity = 2 

CFLP_vals = []
CFLP_LR_vals = []

for t in range(num_trials):

  # Generate data
  customers = [(np.round(np.random.rand(),2), np.round(np.random.rand(),2)) for i in range(num_customers)]  
  facilities = [(np.round(np.random.rand(),2), np.round(np.random.rand(),2)) for i in range(num_facilities)]

  setup_cost = [1.0 for i in range(num_facilities)]
  cost_per_mile = 1.0

  # Compute distances
  def compute_distance(loc1, loc2):
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]  
    return np.sqrt(dx**2 + dy**2)
  
  cartesian_prod = list(product(range(num_customers), range(num_facilities)))
  shipping_cost = {(c,f) : cost_per_mile*compute_distance(customers[c], facilities[f]) for c, f in cartesian_prod}

  # CFLP MIP Model
  m_CFLP = gp.Model("CFLP")
  
  select = m_CFLP.addVars(num_facilities, vtype=GRB.BINARY, name="Select")
  assign = m_CFLP.addVars(cartesian_prod, ub=1, vtype=GRB.CONTINUOUS, name="Assign") 
  
  m_CFLP.addConstrs(assign[(c,f)] <= select[f] for c,f in cartesian_prod)
  m_CFLP.addConstrs(gp.quicksum(assign[(c,f)] for f in range(num_facilities)) == 1 for c in range(num_customers))
  
  # Capacity constraint
  m_CFLP.addConstrs(gp.quicksum(assign[(c,f)] for c in range(num_customers)) <= capacity*select[f] for f in range(num_facilities))
  
  m_CFLP.setObjective(select.prod(setup_cost) + assign.prod(shipping_cost), GRB.MINIMIZE)

  m_CFLP.optimize()
  CFLP_vals.append(m_CFLP.ObjVal)

  # CFLP Linear Relaxation
  m_CFLP_LR = gp.Model("CFLP_LR")
  
  select = m_CFLP_LR.addVars(num_facilities, ub=1, name="Select")
  assign = m_CFLP_LR.addVars(cartesian_prod, ub=1, name="Assign")

  m_CFLP_LR.addConstrs(assign[(c,f)] <= select[f] for c,f in cartesian_prod)
  m_CFLP_LR.addConstrs(gp.quicksum(assign[(c,f)] for f in range(num_facilities)) == 1 for c in range(num_customers))  

  # Capacity constraint
  m_CFLP_LR.addConstrs(gp.quicksum(assign[(c,f)] for c in range(num_customers)) <= capacity*select[f] for f in range(num_facilities))

  m_CFLP_LR.setObjective(select.prod(setup_cost) + assign.prod(shipping_cost), GRB.MINIMIZE)

  m_CFLP_LR.optimize()
  CFLP_LR_vals.append(m_CFLP_LR.ObjVal)

print(len(CFLP_vals), "CFLP optimal values:", CFLP_vals)
print(len(CFLP_LR_vals), "CFLP_LR optimal values:", CFLP_LR_vals)

num_tight = sum(v1 == v2 for v1, v2 in zip(CFLP_vals, CFLP_LR_vals))
print("\nCFLP tight in", num_tight, "trials")