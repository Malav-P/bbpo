import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_shortest_path(W, source, target):
    n, _ = W.shape # (N, N)
    

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    # Create model
    m = gp.Model("shortest_path", env=env)
    m.Params.LogToConsole = 0  # Silence output
    
    # Binary decision variables x[i,j] = 1 if edge i->j is used
    x = m.addMVar(shape=W.shape, vtype=GRB.BINARY, name="x")
    
    # Objective: minimize total cost
    m.setObjective((W*x).sum(), GRB.MINIMIZE)
    
    # Flow conservation constraints
    for i in range(n):
        if i == source:
            m.addConstr(gp.quicksum(x[i,j] for j in range(n)) - gp.quicksum(x[j,i] for j in range(n)) == 1)
        elif i == target:
            m.addConstr(gp.quicksum(x[i,j] for j in range(n)) - gp.quicksum(x[j,i] for j in range(n)) == -1)
        else:
            m.addConstr(gp.quicksum(x[i,j] for j in range(n)) - gp.quicksum(x[j,i] for j in range(n)) == 0)
    
    # Solve
    m.optimize()
    
    # Extract path
    path = []
    # if m.status == GRB.OPTIMAL:
    #     current = source
    #     while current != target:
    #         for j in range(n):
    #             if x[current,j].X > 0.5:
    #                 path.append((current, j))
    #                 current = j
    #                 break

    np.rint(x.X, out=x.X)
    x_var = x.X.astype(int)

    return x_var, path, m.objVal if m.status == GRB.OPTIMAL else None

def solve_shortest_path_time_expanded(W, source, target):
    T, n, _ = W.shape # (T, N, N)
    

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    # Create model
    m = gp.Model("shortest_path", env=env)
    m.Params.LogToConsole = 0  # Silence output
    
    # Binary decision variables x[i,j] = 1 if edge i->j is used
    x = m.addMVar(shape=W.shape, vtype=GRB.BINARY, name="x")
    
    # Objective: minimize total cost
    m.setObjective((W*x).sum(), GRB.MINIMIZE)
    
    # Flow conservation constraints
    for i in range(n):
        if i == source:
            m.addConstr(gp.quicksum(x[:, i,j] for j in range(n)) - gp.quicksum(x[:, j,i] for j in range(n)) == 1)
        elif i == target:
            m.addConstr(gp.quicksum(x[:, i,j] for j in range(n)) - gp.quicksum(x[:, j,i] for j in range(n)) == -1)
        else:
            m.addConstr(gp.quicksum(x[:, i,j] for j in range(n)) - gp.quicksum(x[:, j,i] for j in range(n)) == 0)
    
    # Solve
    m.optimize()
    
    # Extract path
    path = []
    # if m.status == GRB.OPTIMAL:
    #     current = source
    #     while current != target:
    #         for j in range(n):
    #             if x[current,j].X > 0.5:
    #                 path.append((current, j))
    #                 current = j
    #                 break

    np.rint(x.X, out=x.X)
    x_var = x.X.astype(int)

    return x_var, path, m.objVal if m.status == GRB.OPTIMAL else None


