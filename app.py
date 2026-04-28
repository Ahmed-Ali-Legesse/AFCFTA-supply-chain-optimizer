import pandas as pd
import pulp
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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
    base_demand = dict(zip(demand_subset['Country'], demand_subset['gdp_d'].astype(float) * 0.0001))

    coords = {
        'Kenya': (-1.2921, 36.8219), 'Tanzania': (-6.1659, 35.7516),
        'Uganda': (1.3733, 32.2903), 'Rwanda': (-1.9403, 29.8739),
        'Ethiopia': (9.1450, 40.4897)
    }

    return nodes, mfn_tariffs, hurdle_rates, friction_matrix, base_demand, coords

# --- 3. MILP SOLVER ---
def run_milp(nodes, mfn_tariffs, hurdle_rates, friction_matrix, base_demand, roo_compliant, afcfta_phase_down):
    model = pulp.LpProblem("AfCFTA_Pharma_CapEx", pulp.LpMinimize)

    capex_cost = 50.0 
    production_cost = 0.2

    y = pulp.LpVariable.dicts("Hub", nodes, cat='Binary')
    x = pulp.LpVariable.dicts("Route", [(i, j) for i in nodes for j in nodes], lowBound=0, cat='Continuous')

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
    total_ops = pulp.lpSum([x[(i, j)] * (production_cost * friction_matrix.get((i, j), 2.0) * (1 + effective_tariffs[(i, j)])) for i in nodes for j in nodes])
    
    model += total_capex + total_ops

    # Constraints
    for j in nodes:
        model += pulp.lpSum([x[(i, j)] for i in nodes]) >= base_demand[j]
        
    big_m = 1_000_000_000 
    for i in nodes:
        model += pulp.lpSum([x[(i, j)] for j in nodes]) <= y[i] * big_m
        
    model += pulp.lpSum([y[i] for i in nodes]) == 1

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[model.status] != 'Optimal':
        return pulp.LpStatus[model.status], "No Solution", 0.0, 0.0, 0.0, {}

    routing = {(i, j): x[(i, j)].varValue for i in nodes for j in nodes if x[(i, j)].varValue is not None and x[(i, j)].varValue > 0.001}
    hub_list = [i for i in nodes if y[i].varValue is not None and y[i].varValue > 0.5]
    hub = hub_list[0] if hub_list else "Error"
    
    capex_val = capex_cost * (1 + hurdle_rates[hub])
    ops_val = pulp.value(total_ops)

    return pulp.LpStatus[model.status], hub, pulp.value(model.objective), capex_val, ops_val, routing


# --- 4. FRONTEND UI ---
st.title("AfCFTA Pharmaceutical Supply Chain SAA Model")
st.markdown("---")

nodes, mfn_tariffs, hurdle_rates, friction_matrix, base_demand, coords = load_base_parameters()

with st.sidebar:
    st.header("Model Constraints")
    roo_compliant = st.checkbox("AfCFTA Rules of Origin Met (>40% VA)", value=True, help="Toggles the MFN penalty pathway.")
    afcfta_phase_down = st.slider("Tariff Phase-Down Rate", min_value=0.0, max_value=0.05, value=0.01, step=0.01)
    st.markdown("---")
    st.write("**Base Assumptions:**")
    st.write("CapEx: $50M")
    st.write("Target HS: 300490")

status, hub, total_cost, capex_val, ops_val, routing = run_milp(
    nodes, mfn_tariffs, hurdle_rates, friction_matrix, base_demand, roo_compliant, afcfta_phase_down
)

if hub == "No Solution" or status != 'Optimal':
    st.error(f"The solver could not find a valid supply chain network. Status: {status}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Optimal Factory Location", value=hub)
col2.metric(label="Total Network NPV", value=f"${total_cost:.2f}M")
col3.metric(label="Risk-Adjusted CapEx", value=f"${capex_val:.2f}M")
col4.metric(label="Logistics & Tariffs", value=f"${ops_val:.2f}M")
st.markdown("---")

viz_col1, viz_col2 = st.columns([2, 1])

with viz_col1:
    st.subheader("Optimized Trade Routing")
    
    fig_map = go.Figure()
    
    fig_map.add_trace(go.Scattergeo(
        lon=[coords[hub][1]], lat=[coords[hub][0]],
        mode='markers+text', text=[f"{hub} (Hub)"], textposition="bottom center",
        marker=dict(size=14, color='red', symbol='star'), name='Factory'
    ))
    
    for (orig, dest), vol in routing.items():
        if orig != dest:
            fig_map.add_trace(go.Scattergeo(
                lon=[coords[orig][1], coords[dest][1]],
                lat=[coords[orig][0], coords[dest][0]],
                mode='lines', line=dict(width=vol/20, color='blue'),
                opacity=0.6, name=f"To {dest}"
            ))
            fig_map.add_trace(go.Scattergeo(
                lon=[coords[dest][1]], lat=[coords[dest][0]],
                mode='markers+text', text=[dest], textposition="bottom center",
                marker=dict(size=8, color='blue'), name=dest
            ))

    fig_map.update_layout(
        geo_scope='africa',
        geo=dict(center=dict(lat=2.0, lon=35.0), projection_scale=4.5),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    st.plotly_chart(fig_map, use_container_width=True)

with viz_col2:
    st.subheader("Cost Breakdown")
    fig_bar = px.bar(
        x=["CapEx (Risk Adjusted)", "Ops, Tariffs & Friction"], 
        y=[capex_val, ops_val], 
        labels={'x': 'Cost Category', 'y': 'Millions (USD)'},
        color=["CapEx", "Ops"], color_discrete_sequence=['#ef553b', '#636efa']
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Route Volumes")
    route_df = pd.DataFrame([{"Origin": o, "Destination": d, "Volume": v} for (o, d), v in routing.items()])
    st.dataframe(route_df.style.format({"Volume": "{:.1f}"}), hide_index=True)

