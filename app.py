import streamlit as st
import pulp
import numpy as np
import pandas as pd
from collections import Counter
import gc

# --- MACROECONOMIC PROXY DATA (EAST AFRICA TOP 5) ----------------------------------------
hubs = ['Kenya', 'Tanzania', 'Ethiopia', 'Rwanda', 'Uganda']
markets = ['Kenya', 'Tanzania', 'Ethiopia', 'Uganda', 'Rwanda']
years = [1, 2, 3, 4, 5]

# Sovereign Borrowing Rates (Proxy for Cost of Capital 'r')
# Source: Central Bank lending rates / Sovereign Bond Yields
discount_rates = {
    'Rwanda': 0.08,   # High DFI concessional backing
    'Tanzania': 0.10, 
    'Uganda': 0.12, 
    'Kenya': 0.16,    # High domestic debt burden
    'Ethiopia': 0.25  # Severe sovereign default/inflation premium
}

# World Bank Enterprise Surveys - Bureaucratic "Time Tax" 
# Proxy for hidden levies and regulatory operational bleed
# Metric: "Senior management time spent dealing with the requirements of government regulation (%)"
wb_bureaucratic_tax = {
    'Rwanda': 0.04,   # ~4% - Highly digitized and streamlined
    'Uganda': 0.06,   # ~6% 
    'Kenya': 0.09,    # ~9% - Heavy decentralized county-level friction
    'Ethiopia': 0.11, # ~11% - Severe federal/municipal bureaucratic drag
    'Tanzania': 0.12  # ~12% - High regulatory compliance burden
}

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

# Average Transit Days (Perfect Conditions)
transit_days = {
    'Kenya':    {'Kenya': 1, 'Tanzania': 3, 'Ethiopia': 6, 'Uganda': 4, 'Rwanda': 7},
    'Tanzania': {'Kenya': 3, 'Tanzania': 1, 'Ethiopia': 8, 'Uganda': 5, 'Rwanda': 4},
    'Ethiopia': {'Kenya': 6, 'Tanzania': 8, 'Ethiopia': 1, 'Uganda': 7, 'Rwanda': 9},
    'Rwanda':   {'Kenya': 7, 'Tanzania': 4, 'Ethiopia': 9, 'Uganda': 2, 'Rwanda': 1},
    'Uganda':   {'Kenya': 4, 'Tanzania': 5, 'Ethiopia': 7, 'Rwanda': 2, 'Uganda': 1}
}

# --- DYNAMIC DEMAND: THE GRAVITY MODEL ---

# STREAMLIT SIDEBAR
st.sidebar.header("Macroeconomic Policy Shocks")
ethiopia_fx_regime = st.sidebar.radio(
    "Ethiopia FX Regime (Birr Float Impact)",
    ("Pre-Float (2024 Pegged GDP)", "Post-Float (2025 Market GDP)")
)

# The Birr float destroyed Ethiopia's USD-denominated purchasing power
ethiopia_gdp = 150 if ethiopia_fx_regime == "Pre-Float (2024 Pegged GDP)" else 109

gdp_billions = {
    'Kenya': 136, 
    'Tanzania': 87, 
    'Ethiopia': ethiopia_gdp, 
    'Uganda': 66, 
    'Rwanda': 16
}

distance_km = {
    'Kenya':    {'Kenya': 1, 'Tanzania': 800, 'Ethiopia': 1160, 'Uganda': 650, 'Rwanda': 1150},
    'Tanzania': {'Kenya': 800, 'Tanzania': 1, 'Ethiopia': 1750, 'Uganda': 1000, 'Rwanda': 1150},
    'Ethiopia': {'Kenya': 1160, 'Tanzania': 1750, 'Ethiopia': 1, 'Uganda': 1200, 'Rwanda': 1700},
    'Rwanda':   {'Kenya': 1150, 'Tanzania': 1150, 'Ethiopia': 1700, 'Uganda': 500, 'Rwanda': 1},
    'Uganda':   {'Kenya': 650, 'Tanzania': 1000, 'Ethiopia': 1200, 'Rwanda': 500, 'Uganda': 1}
}

def calculate_gravity_demand(origin, destination):
    if origin == destination:
        return gdp_billions[destination]
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
    
# Value of a single unit of goods (e.g., $1,000)
unit_value = 1000 
corporate_wacc = 0.15 # 15% corporate cost of capital
static_mfn_tariff = 25  # Punitive tariff for failing AfCFTA Rules of Origin

# --- THE STOCHASTIC ENGINE -------------------------------------------------------
def run_stochastic_optimizer(volatility_dial, iterations):
    results = Counter()
    
    # UI Elements for memory and UX
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Pre-calculate Amortized CapEx to save loop computing time
    # Formula: F_i * [ r / (1 - (1+r)^-T) ]
    annualized_capex = {}
    for i in hubs:
        r = discount_rates[i]
        annualized_capex[i] = fixed_costs[i] * (r / (1 - (1 + r)**-5))
    
    for step in range(iterations):
        model = pulp.LpProblem(f"Sim_{step}", pulp.LpMinimize)
        
        Y_MFN = pulp.LpVariable.dicts("Hub_MFN", hubs, cat='Binary')
        Y_RoO = pulp.LpVariable.dicts("Hub_RoO", hubs, cat='Binary')
        X_MFN = pulp.LpVariable.dicts("Ship_MFN", [(i, j, t) for i in hubs for j in markets for t in years], lowBound=0)
        X_RoO = pulp.LpVariable.dicts("Ship_RoO", [(i, j, t) for i in hubs for j in markets for t in years], lowBound=0)

        # Force the engine to scale abstract proxy demand up to industrial reality
        volume_multiplier = 10000
        
        total_variable_cost = 0
        for i in hubs:
    # Multiply the base production cost by the World Bank Bureaucratic Time Tax
            cost_mfn_prod = calculate_unit_opex(i, is_roo_compliant=False) * (1 + wb_bureaucratic_tax[i])
            cost_roo_prod = calculate_unit_opex(i, is_roo_compliant=True) * (1 + wb_bureaucratic_tax[i])
            
            for j in markets:
                for t in years:
                    #Asymmetric Exponential Volatility (Chaos only adds friction)
                    sim_freight = base_freight[i][j] * (1 + np.random.exponential(scale=volatility_dial))
                    actual_days = transit_days[i][j] * (1 + np.random.exponential(scale=volatility_dial))
                    #Transit Holding Cost (Capital trapped in logistics)
                    holding_cost = unit_value * corporate_wacc * (actual_days / 365)
                    
                    # 1. Global Sourcing / MFN Path
                    tariff_mfn = 0 if i == j else static_mfn_tariff
                    cost_mfn = (sim_freight + holding_cost + tariff_mfn + cost_mfn_prod) / fx_lambda[j]
                   # MULTIPLY BY VOLUME SCALER
                    total_variable_cost += (cost_mfn * volume_multiplier) * X_MFN[(i, j, t)]
                    
                    # 2. AfCFTA Compliant / RoO Path
                    tariff_roo = 0 if i == j else max(0, 15 * (1 - 0.20 * (t - 1)))
                    cost_roo = (sim_freight + holding_cost + tariff_roo + cost_roo_prod) / fx_lambda[j]
                    total_variable_cost += (cost_roo * volume_multiplier) * X_RoO[(i, j, t)]
                    
                    
       # NEW: Objective function uses Amortized CapEx instead of raw Fixed Costs
        model += pulp.lpSum([(annualized_capex[i] * 1000) * (Y_MFN[i] + Y_RoO[i]) for i in hubs]) + total_variable_cost
        
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

# --- STREAMLIT FRONTEND -----------------------------------------
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
