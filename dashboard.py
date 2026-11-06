import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------
# Page settings
# --------------------------------------------------
st.set_page_config(
    page_title="Next-Day Stock Return Forecasting",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f1420, #111827);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Hide default Streamlit menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main title */
.dashboard-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 0px;
    color: #f4f1ff;
}

.dashboard-subtitle {
    color: #9098b5;
    font-size: 17px;
    margin-bottom: 25px;
}

/* Cards */
.metric-card {
    background: linear-gradient(145deg, #171d2a, #121722);
    border: 1px solid #293144;
    border-radius: 14px;
    padding: 22px;
    min-height: 135px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.20);
}

.metric-label {
    color: #9ca4bd;
    font-size: 14px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 29px;
    font-weight: 700;
    color: #f4f1ff;
}

.metric-change {
    font-size: 13px;
    color: #45d99b;
    margin-top: 5px;
}

/* Panel */
.panel {
    background: linear-gradient(145deg, #171d2a, #121722);
    border: 1px solid #293144;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

.panel-title {
    color: #f4f1ff;
    font-size: 21px;
    font-weight: 600;
    margin-bottom: 3px;
}

.panel-subtitle {
    color: #8e96ae;
    font-size: 14px;
    margin-bottom: 8px;
}

/* Stock button styling */
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 46px;
    border: 1px solid #343d54;
    background-color: #151b28;
    color: #b6bdd0;
    font-weight: 600;
}

div.stButton > button:hover {
    border: 1px solid #8b5cf6;
    color: white;
    background-color: #2f2458;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111723;
    border-right: 1px solid #263044;
}

.sidebar-title {
    font-size: 24px;
    font-weight: 700;
    color: #f2efff;
}

.sidebar-subtitle {
    color: #8f98b5;
    font-size: 13px;
    letter-spacing: 2px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Results
# --------------------------------------------------
results = {
    "AAPL": {
        "Baseline": 0.017037,
        "Linear Regression": 0.017107,
        "Ridge Regression": 0.017045,
        "Random Forest": 0.017040,
        "RNN": 0.017244,
        "direction": 53.66
    },

    "GOOGL": {
        "Baseline": 0.020637,
        "Linear Regression": 0.020765,
        "Ridge Regression": 0.020637,
        "Random Forest": 0.020606,
        "RNN": 0.021071,
        "direction": 51.66
    },

    "NVDA": {
        "Baseline": 0.034473,
        "Linear Regression": 0.034484,
        "Ridge Regression": 0.034411,
        "Random Forest": 0.034395,
        "RNN": 0.034464,
        "direction": 53.79
    },

    "TSLA": {
        "Baseline": 0.038305,
        "Linear Regression": 0.038395,
        "Ridge Regression": 0.038333,
        "Random Forest": 0.038230,
        "RNN": 0.038855,
        "direction": 51.17
    }
}


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.markdown("""
        <div style="padding:15px;">
            <div style="font-size:42px;">📈</div>
            <div class="sidebar-title">Stock Forecast</div>
            <div class="sidebar-title">Dashboard</div>
            <br>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏠 Overview")
    st.markdown("### 📊 Model Analysis")
    st.markdown("### 🗄 Stock Data")

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.markdown("""
        <div class="sidebar-subtitle">
            NEXT-DAY<br>
            STOCK RETURN<br>
            FORECASTING
        </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    '<div class="dashboard-title">Next-Day Stock Return Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Machine learning comparison for next-day stock log-return prediction.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Stock selector
# --------------------------------------------------
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = "GOOGL"

stock_cols = st.columns(4)

stocks = ["AAPL", "GOOGL", "NVDA", "TSLA"]

for i, stock in enumerate(stocks):
    with stock_cols[i]:
        if st.button(stock, key=f"stock_{stock}"):
            st.session_state.selected_stock = stock

stock = st.session_state.selected_stock
stock_data = results[stock]


# --------------------------------------------------
# Determine best trained ML model
# --------------------------------------------------
ml_models = [
    "Linear Regression",
    "Ridge Regression",
    "Random Forest",
    "RNN"
]

best_ml_model = min(
    ml_models,
    key=lambda model: stock_data[model]
)

best_ml_rmse = stock_data[best_ml_model]
baseline_rmse = stock_data["Baseline"]

improvement = (
    (baseline_rmse - best_ml_rmse)
    / baseline_rmse * 100
)


# --------------------------------------------------
# Top cards
# --------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Selected Stock</div>
        <div class="metric-value">{stock}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best ML Model</div>
        <div class="metric-value" style="font-size:23px;">
            {best_ml_model}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best ML RMSE</div>
        <div class="metric-value">{best_ml_rmse:.6f}</div>
        <div class="metric-change">
            {improvement:+.3f}% vs baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Baseline RMSE</div>
        <div class="metric-value">{baseline_rmse:.6f}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# --------------------------------------------------
# RMSE Chart + Direction Gauge
# --------------------------------------------------
left, right = st.columns([2.1, 1])

with left:

    st.markdown("""
    <div class="panel-title">Model Comparison</div>
    <div class="panel-subtitle">
        Root Mean Squared Error (RMSE) across prediction models
    </div>
    """, unsafe_allow_html=True)

    models = [
        "Baseline",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "RNN"
    ]

    values = [stock_data[m] for m in models]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=values,
        text=[f"{x:.6f}" for x in values],
        textposition="outside",
        marker=dict(
            color=[
                "#623bb5",
                "#6c42c7",
                "#7044c5",
                "#935cff",
                "#653ab7"
            ]
        ),
        hovertemplate="%{x}<br>RMSE: %{y:.6f}<extra></extra>"
    ))

    fig.update_layout(
        height=470,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b8bfd3"),
        margin=dict(l=25, r=15, t=30, b=20),
        yaxis_title="RMSE",
        xaxis_title=None,
        showlegend=False
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.10)",
        zeroline=False
    )

    fig.update_xaxes(
        tickfont=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)


with right:

    st.markdown("""
    <div class="panel-title">Direction Accuracy</div>
    <div class="panel-subtitle">
        Correctly predicted next-day direction
    </div>
    """, unsafe_allow_html=True)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stock_data["direction"],
        number={
            "suffix": "%",
            "font": {"size": 37, "color": "#f4f1ff"}
        },

        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 0,
                "tickcolor": "#9098b5"
            },

            "bar": {
                "color": "#8b5cf6"
            },

            "bgcolor": "#262d3a",

            "steps": [
                {
                    "range": [0, 100],
                    "color": "#343b4c"
                }
            ]
        }
    ))

    gauge.update_layout(
        height=325,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#aab2c8"},
        margin=dict(l=30, r=30, t=30, b=20)
    )

    st.plotly_chart(gauge, use_container_width=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title" style="font-size:18px;">
            Prediction Target
        </div>
        <br>
        <b style="color:#9b6cff;">
            Next-Day Log Return
        </b>
        <br><br>
        <span style="color:#9ca4bd;">
            Models are evaluated using chronological
            test data and compared against the zero-return baseline.
        </span>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Summary Table
# --------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="panel-title">Best Results by Stock</div>
<div class="panel-subtitle">
    Lowest RMSE obtained for each stock
</div>
""", unsafe_allow_html=True)


summary = pd.DataFrame({
    "Stock": [
        "AAPL",
        "GOOGL",
        "NVDA",
        "TSLA"
    ],

    "Best Model": [
        "Baseline",
        "Random Forest",
        "Random Forest",
        "Random Forest"
    ],

    "RMSE": [
        0.017037,
        0.020606,
        0.034395,
        0.038230
    ]
})


st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "RMSE": st.column_config.NumberColumn(
            "RMSE",
            format="%.6f"
        )
    }
)


st.markdown("""
<div style="
    color:#858da7;
    font-size:13px;
    margin-top:8px;
">
ⓘ Lower RMSE indicates better predictive performance.
</div>
""", unsafe_allow_html=True)