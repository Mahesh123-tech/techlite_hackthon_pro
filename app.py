import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_data, create_sql_db, run_queries
from models import preprocess, train_models, cluster_analysis

st.set_page_config(page_title="ER Wait-Time Optimizer", layout="wide")

st.title("🏥 Hospital ER Wait-Time Optimizer")

# Upload dataset
file = st.file_uploader("Upload ER dataset CSV", type="csv")

if file:
    df = load_data(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # SQL DATABASE
    conn = create_sql_db(df)
    results = run_queries(conn)

    # ---------------- SQL ANALYSIS ----------------
    st.subheader("📌 SQL Insights")

    st.write("Hourly Wait Time")
    st.line_chart(results['hourly_wait'].set_index('Hour'))

    st.write("Day-wise Wait Time")
    st.bar_chart(results['day_wait'].set_index('Day of Week'))

    # ---------------- HEATMAP ----------------
    st.subheader("🔥 Heatmap (Hour vs Day)")

    heatmap = df.pivot_table(
        values='Total Wait Time (min)',
        index='Day of Week',
        columns='Hour',
        aggfunc='mean'
    )

    fig = px.imshow(heatmap,
                    labels=dict(x="Hour", y="Day", color="Wait Time"),
                    aspect="auto")

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- ML MODELS ----------------
    st.subheader("🤖 ML Prediction Models")

    X, y = preprocess(df)
    lr, rf = train_models(X, y)

    st.success("Models trained successfully!")

    # ---------------- PATIENT INPUT ----------------
    st.subheader("🧑 Patient Wait-Time Prediction")

    urgency = st.slider("Urgency Level (encoded)", 0, 3, 1)
    nurse_ratio = st.slider("Nurse-to-Patient Ratio", 1, 10, 3)
    specialist = st.slider("Specialist Availability", 0, 10, 2)
    beds = st.slider("Facility Beds", 50, 500, 200)
    reg = st.slider("Registration Time", 1, 30, 10)
    triage = st.slider("Triage Time", 1, 30, 10)
    doctor = st.slider("Doctor Wait Time", 1, 60, 20)

    input_data = [[urgency, nurse_ratio, specialist, beds, reg, triage, doctor]]

    if st.button("Predict Wait Time"):
        prediction = rf.predict(input_data)[0]
        st.success(f"Estimated Wait Time: {prediction:.2f} minutes")

    # ---------------- STAFFING SIMULATION ----------------
    st.subheader("👨‍⚕️ Staffing Impact Simulator")

    staff_boost = st.slider("Increase Nurses (%)", 0, 100, 10)

    df['Simulated Wait'] = df['Total Wait Time (min)'] * (1 - staff_boost/100)

    fig2 = px.line(df, y=['Total Wait Time (min)', 'Simulated Wait'],
                   title="Staffing Impact on Wait Time")

    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- CLUSTERING ----------------
    st.subheader("📊 Bottleneck Detection (Clustering)")

    clustered_df, _ = cluster_analysis(df)
    fig3 = px.scatter(clustered_df,
                      x='Nurse-to-Patient Ratio',
                      y='Total Wait Time (min)',
                      color='Cluster')

    st.plotly_chart(fig3, use_container_width=True)

else:
    st.warning("Please upload dataset to start analysis.")
