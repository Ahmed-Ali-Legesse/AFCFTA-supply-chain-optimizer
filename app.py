import streamlit as st
import pulp
import numpy as np
import pandas as pd
from collections import Counter
import gc

# --- MACROECONOMIC PROXY DATA (EAST AFRICA TOP 5) ---
hubs = ['Kenya', 'Tanzania', 'Ethiopia', 'Rwanda', 'Uganda']
markets = ['Kenya', 'Tanzania', 'Ethiopia', 'Uganda', 'Rwanda']
years = [1, 2, 3, 4, 5]

# F_i: Fixed Capex (USD Millions)
fixed_costs = {'Ethiopia': 30, 'Tanzania': 38, 'Uganda': 42, 'Rwanda': 48, 'Kenya': 55}

# K_i: Production Capacity (Units)
capacity = {h: 2000 for h in hubs}

# lambda_i: FX Repatriation Constraints (1.0 = Free flow, 0.5 = 50% trapped)
fx_lambda = {'Kenya': 1.0, 'Tanzania': 0.95, 'Uganda': 1.0, 'Rwanda': 1.0, 'Ethiopia': 0.5}

# C_ij: Transport Friction Baseline (USD per TEU Container / 100)
base_freight = {
    'Kenya':    {'Kenya': 5,  'Uganda': 22, 'Rwanda': 35, 'Tanzania': 15, 'Ethiopia': 40},
    'Tanzania': {'Kenya': 15, 'Uganda': 28, 'Rwanda': 30, 'Tanzania': 5,  'Ethiopia': 45},
    'Ethiopia': {'Kenya': 40, 'Uganda': 50, 'Rwanda': 55, 'Tanzania': 45, 'Ethiopia': 5},
    'Rwanda':   {'Kenya': 35, 'Uganda': 12, 'Rwanda': 3,  'Tanzania': 30, 'Ethiopia': 55},
    'Uganda':   {'Kenya': 22, 'Uganda': 4,  'Rwanda': 12, 'Tanzania': 28, 'Ethiopia': 50}
}
# World Bank LPI Proxy (Converted to a Friction Multiplier)
# Lower is better (1.0 = Frictionless, Higher = Severe Infrastructure Decay)
lpi_friction = {
    'Kenya': 1.2,   # Strong port, SGR rail
    'Rwanda': 1.3,  # Excellent customs efficiency, despite being landlocked
    'Tanzania': 1.5,# Good port, but inland road decay
    'Uganda': 1.8,  # High border friction
    'Ethiopia': 2.2 # Severe customs delays, conflict-prone corridors
}

# --- DYNAMIC DEMAND: THE GRAVITY MODEL ---
gdp_billions = {'Kenya': 136, 'Tanzania': 88, 'Ethiopia': 160, 'Uganda': 66, 'Rwanda': 16}
distance_km = {
    'Kenya':    {'Kenya': 1, 'Tanzania': 800, 'Ethiopia': 1160, 'Uganda': 650, 'Rwanda': 1150},
    'Tanzania': {'Kenya': 800, 'Tanzania': 1, 'Ethiopia': 1750, 'Uganda': 1000, 'Rwanda': 1150},
    'Ethiopia': {'Kenya': 1160, 'Tanzania': 1750, 'Ethiopia': 1, 'Uganda': 1200, 'Rwanda': 1700},
    'Rwanda':   {'Kenya': 1150, 'Tanzania': 1150, 'Ethiopia': 1700, 'Uganda': 500, 'Rwanda': 1},
    'Uganda':   {'Kenya': 650, 'Tanzania': 1000, 'Ethiopia': 1200, 'Rwanda': 500, 'Uganda': 1}
}

def calculate_gravity_demand(origin, destination):
    if origin == destination:
        return gdp_billions[destination] * 2  
        
    # Calculate average corridor friction between the two countries
    corridor_friction = (lpi_friction[origin] + lpi_friction[destination]) / 2
    
    # Effective distance penalizes poor infrastructure
    effective_distance = distance_km[origin][destination] * corridor_friction
    
    return round(100 * (gdp_billions[destination] / effective_distance), 2)

# --- DYNAMIC OPEX: LABOR & ENERGY COMBINED ---
monthly_labor_wage = {'Ethiopia': 35, 'Uganda': 70, 'Rwanda': 90, 'Tanzania': 120, 'Kenya': 160}
power_tariff_kwh = {'Ethiopia': 0.01, 'Tanzania': 0.09, 'Uganda': 0.12, 'Rwanda': 0.19, 'Kenya': 0.22}

def calculate_unit_opex(hub, is_roo_compliant):
    # Proxy: 10 hours of labor + 100 kWh per unit
    labor_cost = (monthly_labor_wage[hub] / 160) * 10
    power_cost = power_tariff_kwh[hub] * 100
    base_opex = labor_cost + power_cost
    
    # 50% premium for forcing localized African supply chains (RoO)
    return base_opex * 1.5 if is_roo_compliant else base_opex

static_mfn_tariff = 25  # Punitive tariff for failing AfCFTA Rules of Origin

# --- THE STOCHASTIC ENGINE ---
def run_stochastic_optimizer(volatility_dial, iterations):
    results = Counter()
    
    # UI Elements for memory and UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(iterations):
        model = pulp.LpProblem(f"Sim_{step}", pulp.LpMinimize)
        
        Y_MFN = pulp.LpVariable.dicts("Hub_MFN", hubs, cat='Binary')
        Y_RoO = pulp.LpVariable.dicts("Hub_RoO", hubs, cat='Binary')
        X_MFN = pulp.LpVariable.dicts("Ship_MFN", [(i, j, t) for i in hubs for j in markets for t in years], lowBound=0)
        X_RoO = pulp.LpVariable.dicts("Ship_RoO", [(i, j, t) for i in hubs for j in markets for t in years], lowBound=0)
        
        total_variable_cost = 0
        for i in hubs:
            cost_mfn_prod = calculate_unit_opex(i, is_roo_compliant=False)
            cost_roo_prod = calculate_unit_opex(i, is_roo_compliant=True)
            
            for j in markets:
                for t in years:
# Base freight is the floor. Volatility only adds cost via an exponential shock.
sim_freight = base_freight[i][j] * (1 + np.random.exponential(scale=volatility_dial))
                    
                    # 1. Global Sourcing / MFN Path
                    tariff_mfn = 0 if i == j else static_mfn_tariff
                    cost_mfn = (sim_freight + tariff_mfn + cost_mfn_prod) / fx_lambda[i]
                    total_variable_cost += cost_mfn * X_MFN[(i, j, t)]
                    
                    # 2. AfCFTA Compliant / RoO Path
                    tariff_roo = 0 if i == j else max(0, 15 * (1 - 0.20 * (t - 1)))
                    cost_roo = (sim_freight + tariff_roo + cost_roo_prod) / fx_lambda[i]
                    total_variable_cost += cost_roo * X_RoO[(i, j, t)]
                    
        # Objective: Unit scale Capex (*1000) + Network Cost
        model += pulp.lpSum([(fixed_costs[i] * 1000) * (Y_MFN[i] + Y_RoO[i]) for i in hubs]) + total_variable_cost
        
        # Constraints
        for i in hubs:
            model += Y_MFN[i] + Y_RoO[i] <= 1  # Mutually exclusive hub types
            
        for j in markets:
            for t in years:
                # Meet Gravity Demand
                route_demand = sum([calculate_gravity_demand(i, j) for i in hubs]) / len(hubs) # Simplified target
                model += pulp.lpSum([X_MFN[(i, j, t)] + X_RoO[(i, j, t)] for i in hubs]) >= route_demand
                
        for i in hubs:
            for t in years:
                model += pulp.lpSum([X_MFN[(i, j, t)] for j in markets]) <= capacity[i] * Y_MFN[i]
                model += pulp.lpSum([X_RoO[(i, j, t)] for j in markets]) <= capacity[i] * Y_RoO[i]
                
        model += pulp.lpSum([Y_MFN[i] + Y_RoO[i] for i in hubs]) >= 1
                
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        active_hubs = []
        for i in hubs:
            if Y_MFN[i].varValue == 1.0: active_hubs.append(f"{i} (Global)")
            elif Y_RoO[i].varValue == 1.0: active_hubs.append(f"{i} (AfCFTA)")
                
        if active_hubs:
            config = " + ".join(sorted(active_hubs))
            results[config] += 1
            
        # RAM Management
        del model
        gc.collect()
        
        # UX Update
        if step % 20 == 0 or step == iterations - 1:
            progress_bar.progress((step + 1) / iterations)
            status_text.text(f"Computing macroeconomic vector {step + 1} of {iterations}...")
            
    status_text.text("Optimization complete.")
    return results

# --- STREAMLIT FRONTEND ---
st.set_page_config(page_title="AfCFTA Stochastic Optimizer", layout="wide")
st.title("AfCFTA Capital Node Optimizer")
st.markdown("Macro-Stochastic MILP balancing Gravity Demand, Volatility, FX Constraints, and RoO Upstream OpEx.")

st.sidebar.header("Stress Test Parameters")
volatility = st.sidebar.slider("Logistics Volatility Index (\u03C3)", 0.0, 0.50, 0.15, 0.05)
iterations = st.sidebar.number_input("Monte Carlo Iterations", min_value=50, max_value=1000, value=250, step=50)

if st.sidebar.button("Run Capital Allocation Engine"):
    outcomes = run_stochastic_optimizer(volatility, iterations)
        
    st.subheader(f"Optimal Network Configurations at {int(volatility*100)}% Macro-Volatility")
    
    if outcomes:
        df = pd.DataFrame.from_dict(outcomes, orient='index', columns=['Frequency']).reset_index()
        df.columns = ['Network Configuration', 'Frequency']
        df['Probability (%)'] = (df['Frequency'] / iterations) * 100
        df = df.sort_values('Probability (%)', ascending=False)
        
        st.bar_chart(data=df, x='Network Configuration', y='Probability (%)', color="#d9534f")
        st.dataframe(df, hide_index=True)
    else:
        st.error("Solver failed to find a feasible configuration under current constraints.")
