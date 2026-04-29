import pandas as pd
import pulp
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AfCFTA Pharma Optimizer", layout="wide", initial_sidebar_state="expanded")

# --- 2. DATA INGESTION ---
@st.cache_data
def load_base_parameters():
    wacc_df = pd.read_csv("data/ea_wacc_parameters.csv")
    gravity_df = pd.read_csv("data/ea_gravity_matrix.csv")
    friction_df = pd.read_csv("data/ea_friction_matrix.csv")

    nodes = ['Kenya', 'Tanzania', 'Uganda', 'Rwanda', 'Ethiopia']

    mfn_tariffs = {n: 0.0 for n in nodes}
    mfn_tariffs['Ethiopia'] = 0.05 

    rf, erp_mature = 0.04, 0.05
    crp_dict = dict(zip(wacc_df['Country'], wacc_df['Country Risk Premium']))
    hurdle_rates = {n: rf + erp_mature + crp_dict.get(n, 0.10) for n in nodes}

    friction_matrix = {}
    for _, row in friction_df.iterrows():
        friction_matrix[(row['Origin'], row['Destination'])] = row['Ad_Valorem_Cost']
    for n in nodes:
        friction_matrix[(n, n)] = 1.0 

    iso_map = {'KEN': 'Kenya', 'TZA': 'Tanzania', 'UGA': 'Uganda', 'RWA': 'Rwanda', 'ETH': 'Ethiopia'}
    gravity_df['Country'] = gravity_df['iso3_d'].map(iso_map)
    
    demand_subset = gravity_df.dropna(subset=['Country', 'gdp_d']).drop_duplicates(subset=['Country'])
    total_gdp = demand_subset['gdp_d'].astype(float).sum()
    
    gdp_weights = {}
    for _, row in demand_subset.iterrows():
        country = row['Country']
        gdp_weights[country] = float(row['gdp_d']) / total_gdp
        
    coords = {
        'Kenya': (-1.2921, 36.8219), 'Tanzania': (-6.1659, 35.7516),
        'Uganda': (1.3733, 32.2903), 'Rwanda': (-1.9403, 29.8739),
        'Ethiopia': (9.1450, 40.4897)
    }

    return nodes, mfn_tariffs, hurdle_rates, friction_matrix, gdp_weights, coords

# --- 3. MILP SOLVER ---
def run_milp(nodes, mfn_tariffs, hurdle_rates, friction_matrix, base_demand, roo_compliant, afcfta_phase_down, selling_price, base_prod_cost, target_volume,friction_multiplier, forced_hub=None):
    model = pulp.LpProblem("AfCFTA_Pharma_5Year_Profit", pulp.LpMaximize)

    capex_cost = 85.0 
    years = ["Year_1", "Year_2", "Year_3", "Year_4", "Year_5"]
    
    production_cost_curve = {
        "Year_1": base_prod_cost + 0.03,
        "Year_2": base_prod_cost + 0.02,
        "Year_3": base_prod_cost + 0.01,
        "Year_4": base_prod_cost,
        "Year_5": base_prod_cost
    }

    y = pulp.LpVariable.dicts("Hub", nodes, cat='Binary')
    x = pulp.LpVariable.dicts("Route", [(i, j, t) for i in nodes for j in nodes for t in years], lowBound=0, cat='Continuous')

    effective_tariffs = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                effective_tariffs[(i, j)] = 0.0
            elif roo_compliant:
                effective_tariffs[(i, j)] = max(0, mfn_tariffs[j] - afcfta_phase_down)
            else:
                effective_tariffs[(i, j)] = mfn_tariffs[j]

    total_capex = pulp.lpSum([y[i] * capex_cost * (1 + hurdle_rates[i]) for i in nodes])
    
    total_ops = pulp.lpSum([
        x[(i, j, t)] * (production_cost_curve[t] * (friction_matrix.get((i, j), 2.0) * friction_multiplier) * (1 + effective_tariffs[(i, j)])) 
        for i in nodes for j in nodes for t in years
    ])
    
    total_revenue = pulp.lpSum([
        x[(i, j, t)] * selling_price 
        for i in nodes for j in nodes for t in years
    ])
    
    model += total_revenue - total_capex - total_ops

    # --- 5-YEAR CONSTRAINTS ---
    min_volume = target_volume 
    max_capacity = 5000.0 

    for t in years:
        for j in nodes:
            model += pulp.lpSum([x[(i, j, t)] for i in nodes]) == base_demand[j]
            
        for i in nodes:
            model += pulp.lpSum([x[(i, j, t)] for j in nodes]) >= y[i] * min_volume
            model += pulp.lpSum([x[(i, j, t)] for j in nodes]) <= y[i] * max_capacity
            
    model += pulp.lpSum([y[i] for i in nodes]) == 1
    
    # 5. FORCE HUB FOR COMPARATIVE ANALYSIS
    if forced_hub:
        model += y[forced_hub] == 1

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[model.status]
    if status != 'Optimal':
        return status, "No Solution", 0, 0, 0, 0, {}

    hub = [i for i in nodes if y[i].varValue and y[i].varValue > 0.5][0]
    
    capex_val = sum([y[i].varValue * capex_cost * (1 + hurdle_rates[i]) for i in nodes])
    ops_val = pulp.value(total_ops)
    rev_val = pulp.value(total_revenue)
    profit_val = rev_val - capex_val - ops_val

    routing = {}
    for i in nodes:
        for j in nodes:
            total_vol = sum(x[(i, j, t)].varValue for t in years if x[(i, j, t)].varValue is not None)
            if total_vol > 0:
                routing[(i, j)] = total_vol / len(years)

    return status, hub, profit_val, rev_val, capex_val, ops_val, routing


# --- 4. FRONTEND UI ---
st.title("AfCFTA Pharmaceutical Supply Chain SAA Model")
st.markdown("---")

with st.sidebar:
    st.header("Model Constraints")
    roo_compliant = st.checkbox("AfCFTA Rules of Origin Met (>40% VA)", value=True)
    afcfta_phase_down = st.slider("Tariff Phase-Down Rate", min_value=0.0, max_value=0.05, value=0.01, step=0.01)
    friction_multiplier = st.slider(
        "Logistics Friction (Border Delays & NTBs)", 
        min_value=1.0, max_value=3.0, value=2.0, step=0.1,
        help="1.0 = Ideal Green-Lane Transit. 3.0 = Massive delays, bribes, and spoilage.")
    st.markdown("---")
    st.header("Unit Economics")
    selling_price = st.slider("Wholesale Selling Price per pill ($)", min_value=0.05, max_value=0.50, value=0.21, step=0.01)
    base_prod_cost = st.slider("Target Production Cost per pill ($)", min_value=0.03, max_value=0.20, value=0.07, step=0.01)
    target_volume = st.slider("Target Annual Volume (Millions)", min_value=50.0, max_value=600.0, value=300.0, step=10.0)
    st.markdown("---")
    st.write("**Base Assumptions:**")
    st.write("CapEx: $85M (WHO-GMP Compliant)")
    st.write("Target HS: 300490")
    
# Load the cached baseline data 
nodes, mfn_tariffs, hurdle_rates, friction_matrix, gdp_weights, coords = load_base_parameters()

# Dynamically apply the slider volume to the cached weights
base_demand = {country: weight * target_volume for country, weight in gdp_weights.items()}


# --- THE SILENT RUN & DROPDOWN ARCHITECTURE ---

# 1. Run the solver silently without forcing a hub to find the true mathematical optimum
_, true_optimal_hub, _, _, _, _, _ = run_milp(
    nodes, mfn_tariffs, hurdle_rates, friction_matrix, base_demand, 
    roo_compliant, afcfta_phase_down, selling_price, base_prod_cost, target_volume,
    friction_multiplier,
    forced_hub=None
)

# 2. Announce the winner, but give the user control to simulate any country
st.success(f"🏆 Algorithm Recommendation: **{true_optimal_hub}** is mathematically the most cost-effective hub location.")

selected_hub = st.selectbox(
    "Select a country below to simulate the factory network and view its specific financial impact:", 
    options=nodes, 
    index=nodes.index(true_optimal_hub) # Automatically defaults to the true optimal winner
)

# 3. Rerun the solver forcing their selected hub
status, hub, profit_val, rev_val, capex_val, ops_val, routing = run_milp(
    nodes, mfn_tariffs, hurdle_rates, friction_matrix, base_demand, 
    roo_compliant, afcfta_phase_down, selling_price, base_prod_cost, target_volume,
    friction_multiplier,
    forced_hub=selected_hub
)

if hub == "No Solution" or status != 'Optimal':
    st.error(f"The solver could not build a viable network from {selected_hub}.")
    st.stop()


# --- THE DASHBOARD METRICS ---

# Calculate the Selected Hub's Break-Even Price 
five_year_volume = target_volume * 5.0
hub_bep = (capex_val + ops_val) / five_year_volume

# --- ROW 1: The Executive Summary ---
row1_col1, row1_col2, row1_col3 = st.columns(3)
row1_col1.metric(label="Simulated Hub", value=hub)
row1_col2.metric(label="Break-Even Price", value=f"${hub_bep:.3f}")

# Turn Profit red if it operates at a deficit
delta_color = "normal" if profit_val > 0 else "inverse"
row1_col3.metric(label="5-Yr Net Profit", value=f"${profit_val:,.1f}M", delta="Deficit" if profit_val < 0 else None, delta_color=delta_color)

# --- Add a small visual gap between the rows ---
st.write("") 

# --- ROW 2: The Financial Mechanics ---
row2_col1, row2_col2, row2_col3 = st.columns(3)
row2_col1.metric(label="Gross Rev", value=f"${rev_val:,.1f}M")
row2_col2.metric(label="Risk Adjusted CapEx", value=f"${capex_val:,.1f}M")
row2_col3.metric(label="5-Yr OpEx", value=f"${ops_val:,.1f}M")

st.markdown("---")


# --- THE VISUALIZATIONS ---

viz_col1, viz_col2 = st.columns([2, 1])

with viz_col1:
    st.subheader("Simulated Trade Routing")
    
    max_route_vol = max([vol for (u, v), vol in routing.items() if vol > 0], default=1)
    hub_lat, hub_lon = coords[hub]
    m = folium.Map(location=[hub_lat, hub_lon], zoom_start=4, tiles="CartoDB positron")

    for (u, v), vol in routing.items():
        if vol > 0.1 and u != v:  
            start_loc = coords[u]
            end_loc = coords[v]
            
            line_weight = max(2.0, (vol / max_route_vol) * 10.0)
            
            folium.PolyLine(
                locations=[start_loc, end_loc],
                weight=line_weight,
                color="#004c6d",  
                opacity=0.7,
                tooltip=f"<b>Route:</b> {u} ➔ {v}<br><b>Volume:</b> {vol:,.1f} Million Units"
            ).add_to(m)

    for node, coord in coords.items():
        if node == hub:
            folium.Marker(
                location=coord,
                tooltip=f"<b>MANUFACTURING HUB: {node}</b><br>WHO-GMP CapEx: $85M",
                icon=folium.Icon(color="red", icon="star", prefix="fa")
            ).add_to(m)
        else:
            received_vol = sum([vol for (u, v), vol in routing.items() if v == node])
            if received_vol > 0:
                folium.CircleMarker(
                    location=coord,
                    radius=6,
                    color="#00A36C",
                    fill=True,
                    fillOpacity=1.0,
                    tooltip=f"<b>Target Market:</b> {node}<br><b>Inbound:</b> {received_vol:,.1f} Million Units"
                ).add_to(m)

    st_folium(m, use_container_width=True, height=500)

with viz_col2:
    st.subheader("5-Year Financial P&L")
    
    fig_bar = px.bar(
        x=["Gross Rev", "OpEx", "CapEx", "Net Profit"], 
        y=[rev_val, ops_val, capex_val, profit_val], 
        labels={'x': '', 'y': 'Millions (USD)'},
        color=["Gross Rev", "OpEx", "CapEx", "Net Profit"], 
        color_discrete_sequence=['#2ca02c', '#636efa', '#ef553b', '#ff7f0e']
    )
    fig_bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Route Volumes")
    route_df = pd.DataFrame([{"Origin": o, "Destination": d, "Volume": v} for (o, d), v in routing.items() if v > 0.1])
    st.dataframe(route_df.style.format({"Volume": "{:,.1f}M"}), hide_index=True, use_container_width=True)
