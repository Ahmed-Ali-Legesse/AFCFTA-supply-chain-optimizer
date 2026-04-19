import streamlit as st
import pulp
import numpy as np
import pandas as pd
from collections import Counter
# --- PROXY DATA (EAST AFRICA TOP 5) --------------------------------------------------------
hubs = ['Kenya', 'Tanzania', 'Ethiopia', 'Rwanda', 'Uganda']
markets = ['Kenya', 'Tanzania', 'Ethiopia', 'Uganda', 'Rwanda']
years = [1, 2, 3, 4, 5]
# F_i: Fixed Capex (USD Millions)
# Baseline $30M, scaled by Industrial Electricity Tariffs (kWh) as an operational proxy.
# Source: GlobalPetrolPrices / Africa Data Hub (2025/2026 data)
fixed_costs = {
    'Ethiopia': 30, # ~$0.01/kWh (Hydro-subsidized)
    'Tanzania': 38, # ~$0.09/kWh 
    'Uganda': 42,   # ~$0.12/kWh
    'Rwanda': 48,   # ~$0.19/kWh
    'Kenya': 55     # ~$0.22/kWh (High thermal/tax burden)
}
# D_j: Demand Weighting (Proxy: World Bank Nominal GDP in USD Billions)
demand = {
    'Ethiopia': 150, 
    'Kenya': 110, 
    'Tanzania': 75, 
    'Uganda': 45, 
    'Rwanda': 13
}
# K_i: Production Capacity (Scaled to handle peak regional demand)
capacity = {h: 500 for h in hubs}
# C_ij: Transport Friction (USD per TEU Container / 100 to normalize with GDP demand scale)
# Source: Northern Corridor Transit and Transport Coordination Authority (NCTTCA) & TradeMark Africa
base_freight = {
    'Kenya':    {'Kenya': 5,  'Uganda': 22, 'Rwanda': 35, 'Tanzania': 15, 'Ethiopia': 40},
    'Tanzania': {'Kenya': 15, 'Uganda': 28, 'Rwanda': 30, 'Tanzania': 5,  'Ethiopia': 45},
    'Ethiopia': {'Kenya': 40, 'Uganda': 50, 'Rwanda': 55, 'Tanzania': 45, 'Ethiopia': 5},
    'Rwanda':   {'Kenya': 35, 'Uganda': 12, 'Rwanda': 3,  'Tanzania': 30, 'Ethiopia': 55},
    'Uganda':   {'Kenya': 22, 'Uganda': 4,  'Rwanda': 12, 'Tanzania': 28, 'Ethiopia': 50}
}

# lambda_i: FX Repatriation Constraints (1.0 = Free flow, 0.5 = 50% trapped)
# Source: IMF / Central Bank Directives (e.g., NBE 50% export surrender rule)
fx_lambda = {
    'Kenya': 1.0, 
    'Tanzania': 0.95, # Minor dollar rationing friction
    'Uganda': 1.0, 
    'Rwanda': 1.0, 
    'Ethiopia': 0.5   # Severe physical goods export capital trap
}
# --- THE STOCHASTIC ENGINE --------------------------------------------------------------------
def run_stochastic_optimizer(volatility_dial, iterations=100):
    results = Counter()
    
    for _ in range(iterations):
        model = pulp.LpProblem("AfCFTA_Sensitivity", pulp.LpMinimize)
        Y = pulp.LpVariable.dicts("Hub", hubs, cat='Binary')
        X = pulp.LpVariable.dicts("Ship", [(i, j, t) for i in hubs for j in markets for t in years], lowBound=0)
        
        total_variable_cost = 0
        for i in hubs:
            for j in markets:
                for t in years:
                    #  Floor the variance at 0 to prevent the solver farming negative freight costs
                    sim_freight = max(0, np.random.normal(base_freight[i][j], base_freight[i][j] * volatility_dial))
                    
                    tariff = 0 if i == j else max(0, 15 * (1 - 0.20 * (t - 1)))
                    risk_adjusted_cost = (sim_freight + tariff) / fx_lambda[i]
                    total_variable_cost += risk_adjusted_cost * X[(i, j, t)]
                    
        model += pulp.lpSum([fixed_costs[i] * Y[i] for i in hubs]) + total_variable_cost
        
        for j in markets:
            for t in years:
                model += pulp.lpSum([X[(i, j, t)] for i in hubs]) == demand[j]
        for i in hubs:
            for t in years:
                model += pulp.lpSum([X[(i, j, t)] for j in markets]) <= capacity[i] * Y[i]
                
        model += pulp.lpSum([Y[i] for i in hubs]) >= 1
                
        model.solve(pulp.PULP_CBC_CMD(msg=False))
  
        # Check for optimality and use > 0.5 to dodge floating point solver inaccuracies
        if pulp.LpStatus[model.status] == 'Optimal':
            config = " + ".join(sorted([i for i in hubs if Y[i].varValue is not None and Y[i].varValue > 0.5]))
            results[config] += 1
        else:
            results["Infeasible Scenario"] += 1      
    return results
  # --- STREAMLIT FRONTEND ----------------------------------------------------------------------------------
st.set_page_config(page_title="AfCFTA Optimizer", layout="wide")
st.title("AfCFTA Capital Node Optimizer")
st.markdown("Macro-Stochastic supply chain routing across the top 5 East African economies, anchored on NCTTCA freight data and World Bank proxies.")

st.sidebar.header("Stress Test Parameters")
volatility = st.sidebar.slider("Macro-Volatility Index (Logistics Friction Variance)", 0.0, 0.50, 0.10, 0.05)
iterations = st.sidebar.number_input("Monte Carlo Iterations", min_value=10, max_value=500, value=50, step=10)
if st.sidebar.button("Run Capital Allocation Engine"):
    with st.spinner("Solving multi-year MILP constraints under stochastic variance..."):
        outcomes = run_stochastic_optimizer(volatility, iterations)      
    st.subheader(f"Optimal Network Configurations at {int(volatility*100)}% Logistics Volatility")
    df = pd.DataFrame.from_dict(outcomes, orient='index', columns=['Frequency']).reset_index()
    df.columns = ['Network Configuration', 'Frequency']
    df['Probability (%)'] = (df['Frequency'] / iterations) * 100
    df = df.sort_values('Probability (%)', ascending=False)  
    st.bar_chart(data=df.set_index('Network Configuration')['Probability (%)'], color="#d9534f")
    st.dataframe(df, hide_index=True) 
    #  Finish UI
    st.info("Analytical Note: Configurations with higher stability across Monte Carlo iterations represent robust capital deployment strategies against regional logistics friction.")
