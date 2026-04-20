import streamlit as st
import pulp
import numpy as np
import pandas as pd
import gc

# --- 1. MACROECONOMIC & FINANCIAL DATA ---
hubs = ['Kenya', 'Tanzania', 'Ethiopia', 'Rwanda', 'Uganda']
markets = ['Kenya', 'Tanzania', 'Ethiopia', 'Uganda', 'Rwanda']
years = [1, 2, 3, 4, 5]

# Corporate Finance: WACC Components
risk_free_rate = 0.04
equity_premium = 0.06
corporate_tax_rate = {h: 0.30 for h in hubs}

# UPGRADE 1: Dynamic Capital Structure based on local credit market depth
# Ethiopia has severe liquidity crunches; Kenya has deep domestic debt markets.
debt_capacity = {'Kenya': 0.50, 'Tanzania': 0.40, 'Uganda': 0.35, 'Rwanda': 0.30, 'Ethiopia': 0.10}

# Hub-specific risk profiles
beta = {'Kenya': 1.1, 'Tanzania': 1.2, 'Uganda': 1.3, 'Rwanda': 1.0, 'Ethiopia': 1.8}
crp = {'Kenya': 0.08, 'Tanzania': 0.06, 'Uganda': 0.07, 'Rwanda': 0.04, 'Ethiopia': 0.18}
local_debt_rate = {'Kenya': 0.14, 'Tanzania': 0.11, 'Uganda': 0.13, 'Rwanda': 0.09, 'Ethiopia': 0.22}

# Calculate adjusted WACC per hub
wacc = {}
for h in hubs:
    cost_of_equity = risk_free_rate + (beta[h] * equity_premium) + crp[h]
    cost_of_debt = local_debt_rate[h] * (1 - corporate_tax_rate[h])
    w_d = debt_capacity[h]
    w_e = 1 - w_d
    wacc[h] = (w_e * cost_of_equity) + (w_d * cost_of_debt)

# Bureaucratic Time Tax & CapEx
time_tax = {'Rwanda': 0.04, 'Uganda': 0.06, 'Kenya': 0.09, 'Ethiopia': 0.11, 'Tanzania': 0.12}
fixed_costs = {'Ethiopia': 30, 'Tanzania': 38, 'Uganda': 42, 'Rwanda': 48, 'Kenya': 55}
capacity = {h: 2000 for h in hubs}
fx_capacity = {'Kenya': 5000, 'Tanzania': 4000, 'Uganda': 3000, 'Rwanda': 2000, 'Ethiopia': 250}

lpi_friction = {'Rwanda': 1.3, 'Kenya': 1.2, 'Tanzania': 1.5, 'Uganda': 1.8, 'Ethiopia': 2.2}
base_freight = {
    'Kenya':    {'Kenya': 0.005, 'Uganda': 0.022, 'Rwanda': 0.035, 'Tanzania': 0.015, 'Ethiopia': 0.040},
    'Tanzania': {'Kenya': 0.015, 'Tanzania': 0.005, 'Ethiopia': 0.045, 'Uganda': 0.028, 'Rwanda': 0.030},
    'Ethiopia': {'Kenya': 0.040, 'Tanzania': 0.045, 'Ethiopia': 0.005, 'Uganda': 0.050, 'Rwanda': 0.055},
    'Rwanda':   {'Kenya': 0.035, 'Tanzania': 0.030, 'Ethiopia': 0.055, 'Uganda': 0.012, 'Rwanda': 0.003},
    'Uganda':   {'Kenya': 0.022, 'Tanzania': 0.028, 'Ethiopia': 0.050, 'Rwanda': 0.012, 'Uganda': 0.004}
}

# --- 2. DGP: PPML GRAVITY DEMAND & BOM ---
gdp_billions = {'Kenya': 136, 'Tanzania': 87, 'Ethiopia': 109, 'Uganda': 66, 'Rwanda': 16}
distance_km = {
    'Kenya':    {'Kenya': 1, 'Tanzania': 800, 'Ethiopia': 1160, 'Uganda': 650, 'Rwanda': 1150},
    'Tanzania': {'Kenya': 800, 'Tanzania': 1, 'Ethiopia': 1750, 'Uganda': 1000, 'Rwanda': 1150},
    'Ethiopia': {'Kenya': 1160, 'Tanzania': 1750, 'Ethiopia': 1, 'Uganda': 1200, 'Rwanda': 1700},
    'Rwanda':   {'Kenya': 1150, 'Tanzania': 1150, 'Ethiopia': 1700, 'Uganda': 500, 'Rwanda': 1},
    'Uganda':   {'Kenya': 650, 'Tanzania': 1000, 'Ethiopia': 1200, 'Rwanda': 500, 'Uganda': 1}
}

# UPGRADE 2: PPML Gravity Model Proxy handling Zero-Trade flows
def get_ppml_demand(origin, destination, year):
    if origin == destination: 
        return gdp_billions[destination] * 2 * (1.03 ** year) 
    
    # Structural elasticities
    beta_0 = 5.0
    beta_1 = 0.8 # Origin mass
    beta_2 = 0.8 # Destination mass
    beta_3 = 1.2 # Distance decay
    
    # Probability of zero trade (Extensive Margin)
    # Scales with distance and destination friction
    friction_index = (distance_km[origin][destination] / 1000) * lpi_friction[destination]
    zero_trade_prob = max(0, min(1, (friction_index - 1.5) / 2))
    
    if np.random.uniform(0, 1) < zero_trade_prob:
        return 0.0 # Corridor is practically dead
        
    mass = (gdp_billions[origin] ** beta_1) * (gdp_billions[destination] ** beta_2)
    friction = (distance_km[origin][destination] ** beta_3) * lpi_friction[destination]
    
    expected_trade = np.exp(beta_0) * (mass / friction)
    return expected_trade * (1.03 ** year)

bom_imported = 0.50 
bom_local = 0.65    

# --- 3. ENTERPRISE OPTIMIZATION ENGINE ---
def run_dynamic_saa_model(volatility, n_scenarios):
    npv_capex = {i: fixed_costs[i] for i in hubs}
    
    # Generate Log-Normal Scenarios
    scenarios = {}
    np.random.seed(42) # Seed for UI stability, remove in production
    for s in range(n_scenarios):
        scenarios[s] = {}
        for j in markets:
            sigma = volatility * lpi_friction[j]
            mu = -0.5 * (sigma ** 2) 
            scenarios[s][j] = np.random.lognormal(mean=mu, sigma=sigma)

    # UPGRADE 3: Pre-compute demand to avoid loop overhead
    demand_matrix = {}
    for s in range(n_scenarios):
        for t in years:
            for j in markets:
                base_d = sum([get_ppml_demand(i, j, t) for i in hubs]) / len(hubs)
                demand_matrix[(j, t, s)] = base_d / scenarios[s][j]

    model = pulp.LpProblem("AfCFTA_Enterprise_SAA", pulp.LpMinimize)
    
    Y_MFN = pulp.LpVariable.dicts("Hub_MFN", hubs, cat='Binary')
    Y_RoO = pulp.LpVariable.dicts("Hub_RoO", hubs, cat='Binary')
    
    # Vectorized dictionary generation for memory efficiency
    indices = [(i, j, t, s) for i in hubs for j in markets for t in years for s in range(n_scenarios)]
    X_MFN = pulp.LpVariable.dicts("Ship_MFN", indices, lowBound=0)
    X_RoO = pulp.LpVariable.dicts("Ship_RoO", indices, lowBound=0)
    
    # Objective Construction
    expected_npv_opex = []
    for s in range(n_scenarios):
        for t in years:
            for i in hubs:
                discount_factor = (1 + wacc[i]) ** t
                for j in markets:
                    freight_cost = base_freight[i][j] * scenarios[s][j]
                    
                    tariff_mfn = 0 if i == j else 0.25 * bom_imported
                    tariff_roo = 0 if i == j else max(0, 0.15 - (0.03 * t)) * bom_local
                    
                    cost_mfn = (bom_imported + freight_cost + tariff_mfn) * (1 + time_tax[i])
                    cost_roo = (bom_local + freight_cost + tariff_roo) * (1 + time_tax[i])
                    
                    expected_npv_opex.append((cost_mfn / discount_factor / n_scenarios) * X_MFN[(i, j, t, s)])
                    expected_npv_opex.append((cost_roo / discount_factor / n_scenarios) * X_RoO[(i, j, t, s)])
                    
    capex_total = pulp.lpSum([npv_capex[i] * (Y_MFN[i] + Y_RoO[i]) for i in hubs])
    model += capex_total + pulp.lpSum(expected_npv_opex)

    # Constraints
    for i in hubs:
        model += Y_MFN[i] + Y_RoO[i] <= 1 

    for s in range(n_scenarios):
        for t in years:
            for j in markets:
                # 1. Demand Satisfaction
                model += pulp.lpSum([X_MFN[(i, j, t, s)] + X_RoO[(i, j, t, s)] for i in hubs]) >= demand_matrix[(j, t, s)]
                
                # 2. Hard FX Constraint
                fx_drain = []
                for i in hubs:
                    if i != j:
                        f_cost = base_freight[i][j] * scenarios[s][j]
                        fx_drain.append((bom_imported + f_cost) * X_MFN[(i, j, t, s)])
                        fx_drain.append((bom_local + f_cost) * X_RoO[(i, j, t, s)])
                if fx_drain:
                    model += pulp.lpSum(fx_drain) <= fx_capacity[j]

    for s in range(n_scenarios):
        for t in years:
            for i in hubs:
                # 3. Capacity Constraints
                model += pulp.lpSum([X_MFN[(i, j, t, s)] for j in markets]) <= capacity[i] * Y_MFN[i]
                model += pulp.lpSum([X_RoO[(i, j, t, s)] for j in markets]) <= capacity[i] * Y_RoO[i]

    model += pulp.lpSum([Y_MFN[i] + Y_RoO[i] for i in hubs]) >= 1

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Cleanup for memory
    del expected_npv_opex
    gc.collect()
    
    if pulp.LpStatus[model.status] == 'Optimal':
        config = []
        for i in hubs:
            if Y_MFN[i].varValue == 1.0: config.append(f"{i} (Global Sourcing MFN)")
            elif Y_RoO[i].varValue == 1.0: config.append(f"{i} (AfCFTA Regional RoO)")
        return pulp.value(model.objective), config
    return None, None

# --- 4. STREAMLIT FRONTEND ---
st.set_page_config(page_title="AfCFTA CapEx Optimizer", layout="wide")
st.title("AfCFTA Multi-Period Stochastic Optimizer (PPML Edition)")
st.markdown("Integrates deep credit WACC, PPML zero-trade DGP, Fat-Tailed Scenarios, and Policy Phase-Downs.")

c1, c2 = st.sidebar.columns(2)
volatility = st.sidebar.slider("Log-Normal Volatility (\u03C3)", 0.0, 0.80, 0.20, 0.05)
n_scenarios = st.sidebar.number_input("SAA Scenarios", min_value=10, max_value=200, value=25, step=5)

if st.sidebar.button("Execute Capital Allocation"):
    with st.spinner(f'Solving Multi-Period Matrix with {n_scenarios} fat-tailed scenarios...'):
        obj_val, config = run_dynamic_saa_model(volatility, n_scenarios)
    
    if config:
        st.success("Mathematical Optimum Located")
        st.metric(label="Total Expected NPV (CapEx + 5yr OpEx, USD Millions)", value=f"${obj_val:,.2f}")
        st.write("### Recommended Capital Deployment:")
        for c in config:
            st.markdown(f"- **{c}**")
    else:
        st.error("Infeasible. FX constraints breached under severe scenario shocks. Cannot fulfill demand.")
