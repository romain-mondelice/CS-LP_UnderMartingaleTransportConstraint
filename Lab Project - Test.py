import cvxpy as cp
import numpy as np
from scipy.stats import norm

"""
------------------------------------------------------------------------------------------------
- Functions
------------------------------------------------------------------------------------------------
"""
def black_scholes_call_price(S0, K, T, r, sigma, q=0):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def cost_function(S_max, S_T):
    return max(S_max - K, 0) - max(S_T - K, 0)

"""
------------------------------------------------------------------------------------------------
- General parameters
------------------------------------------------------------------------------------------------
"""
T = 3/12  # 3 months
dt = 1/252
time_steps = int(T/dt)
S0 = 4000
sigma = 0.2
r = 0.01
K = 4000

# Simulate price paths using geometric Brownian motion
np.random.seed(42)
price_paths = [S0]
for t in range(1, time_steps + 1):
    dW = np.sqrt(dt) * np.random.normal()
    dS = r * price_paths[t-1] * dt + sigma * price_paths[t-1] * dW
    price_paths.append(price_paths[t-1] + dS)

price_domain = np.linspace(0.9 * min(price_paths), 1.1 * max(price_paths), time_steps)

"""
------------------------------------------------------------------------------------------------
- Setup: Linear programming problem
------------------------------------------------------------------------------------------------
"""
P = cp.Variable((time_steps, time_steps), nonneg=True)
costs = np.array([[cost_function(S_max, S_T) for S_T in price_domain] for S_max in price_domain])

### CONSTRAINTS
# Martingale constraint
# Here we try with a strict martingale but got unfeasible problem --> So we relaxed the constraint
strict_martingale_constraints = [cp.sum(cp.multiply(cp.diag(P), price_domain)) == price_domain]

epsilon = cp.Variable()
epsilon_value = 1000
relaxed_martingale_constraints = [cp.sum(cp.multiply(cp.diag(P), price_domain)) >= price_domain - epsilon,
                                  cp.sum(cp.multiply(cp.diag(P), price_domain)) <= price_domain + epsilon,
                                  epsilon >= 0,
                                  epsilon <= epsilon_value]

# Marginal constraint
marginal_constraints = [cp.sum(P, axis=1) == 1, cp.sum(P, axis=0) == 1]

### COST FUNCTION
cost_function = cp.sum(cp.multiply(P, costs))

### BREGMAN PROJECTION
iterations = 10
epsilon = 1e-5
for j in range(iterations):
    objective = cp.Maximize(cp.sum(cp.multiply(P, costs)))
    problem = cp.Problem(objective, marginal_constraints + relaxed_martingale_constraints)
    problem.solve(solver=cp.SCS)
    if problem.status != "optimal":
        print(f"Solver status at iteration {j}: {problem.status}")
        break
    # Update the cost matrix based on Bregman projection directly with CVXPY operations
    costs += epsilon * cp.log(P + 1e-10).value


"""
------------------------------------------------------------------------------------------------
- Results
------------------------------------------------------------------------------------------------
"""

#Here we check the difference, if we respect well or not the martingale constraint
LHS = P.value @ price_domain
RHS = price_domain
#print("LHS:", LHS[:10])
#print("RHS:", RHS[:10])

#Classic call option price
call_option_price = black_scholes_call_price(S0, K, T, r, sigma)
#Get the maximum spread --> our objective function
lookback_option_spread = problem.value

print("Call option price premium: ", round(call_option_price, 2))
print("Difference between premiums of basic options and exotic lookback option: ", round(lookback_option_spread, 2))
print("Final exotic lookback option price: ", round(call_option_price+lookback_option_spread, 2))
