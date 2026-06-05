import streamlit as st
import pandas as pd
import plotly.express as px

from utils import create_sql_db, run_queries
from models import preprocess, train_models, cluster_analysis

st.set_page_config(page_title="Hospital Emergency Dataset Analysis", layout="wide")

st.title("🏥 Hospital Emergency Dataset Analysis Dashboard")

# Upload CSV (NOT JSON)
file = st.file_uploader("Upload Hospital ER_Data.csv", type="csv")

if file:

    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # SQL DB
    conn = create_sql_db(df)
    results = run_queries(conn)

    # SQL Insights
    st.subheader("📌 ER Bottleneck Analysis")

    st.line_chart(results['hourly_wait'].set_index('Hour'))
    st.bar_chart(results['day_wait'].set_index('Day of Week'))

    # Heatmap
    st.subheader("🔥 Wait Time Heatmap")

    heatmap = df.pivot_table(
        values='Total Wait Time (min)',
        index='Day of Week',
        columns='Hour',
        aggfunc='mean'
    )

    fig = px.imshow(heatmap, aspect="auto", title="ER Congestion Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    # ML Models
    st.subheader("🤖 Prediction System")

    X, y = preprocess(df)
    lr, rf = train_models(X, y)

    st.success("Models trained successfully")

    # Input
    urgency = st.slider("Urgency Level", 0, 3, 1)
    nurse_ratio = st.slider("Nurse Ratio", 1, 10, 3)
    specialist = st.slider("Specialist Availability", 0, 10, 2)
    beds = st.slider("Beds", 50, 500, 200)
    reg = st.slider("Registration Time", 1, 30, 10)
    triage = st.slider("Triage Time", 1, 30, 10)
    doctor = st.slider("Doctor Wait Time", 1, 60, 20)

    input_data = [[urgency, nurse_ratio, specialist, beds, reg, triage, doctor]]

    if st.button("Predict Wait Time"):
        pred = rf.predict(input_data)[0]
        st.success(f"Estimated Wait Time: {pred:.2f} minutes")

    # Simulation
    st.subheader("👨‍⚕️ Staffing Simulation")

    boost = st.slider("Increase Staff (%)", 0, 100, 10)
    df['Simulated Wait'] = df['Total Wait Time (min)'] * (1 - boost/100)

    fig2 = px.line(df, y=['Total Wait Time (min)', 'Simulated Wait'])
    st.plotly_chart(fig2)

    # Clustering
    st.subheader("📊 Bottleneck Clusters")

    clustered_df, _ = cluster_analysis(df)

    fig3 = px.scatter(
        clustered_df,
        x='Nurse-to-Patient Ratio',
        y='Total Wait Time (min)',
        color='Cluster'
    )

    st.plotly_chart(fig3)

else:
    st.warning("Please upload Hospital ER_Data.csv to continue")
