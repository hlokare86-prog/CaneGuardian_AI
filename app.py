import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Page Config
st.set_page_config(
    page_title="CaneGuardian AI",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.metric-card {
    background: linear-gradient(135deg, #4CAF50, #2E7D32);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
}
.big-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2E7D32;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="big-title">🌾 CaneGuardian AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smart Farming Monitoring & Decision System</div>', unsafe_allow_html=True)

# Load Data
df = pd.read_csv("cane_guardian_daily_dataset.csv")
latest = df.iloc[-1]

# Sidebar
st.sidebar.title("⚙ Farm Control Panel")

selected_day = st.sidebar.slider(
    "Select Day",
    0,
    len(df)-1,
    len(df)-1
)

selected_data = df.iloc[selected_day]

# Top Metrics
st.markdown("## 📊 Live Farm Status")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🌡 Temperature", f"{selected_data['Temp_C']} °C")
col2.metric("💧 Soil Moisture", f"{selected_data['Soil_Moisture_%']} %")
col3.metric("🌿 Plant Height", f"{selected_data['Plant_Height_cm']} cm")
col4.metric("🦠 Disease Score", f"{selected_data['Disease_Score']}")

# AI Recommendations
st.markdown("## 🤖 AI Recommendations")

col5, col6, col7 = st.columns(3)

# Water Logic
with col5:
    if selected_data["Soil_Moisture_%"] < 38:
        st.error("🚨 Water Needed")
    else:
        st.success("✅ Water Level Good")

# Fertilizer Logic
with col6:
    if selected_data["Nitrogen"] < 140:
        st.warning("⚠ Add Fertilizer")
    else:
        st.success("✅ Nutrients Balanced")

# Spray Logic
with col7:
    if selected_data["Disease_Score"] >= 2:
        st.error("🚨 Spray Needed")
    else:
        st.success("✅ No Spray Required")

# Health Score
st.markdown("## ❤️ Crop Health Score")

health_score = (
    selected_data["Leaf_Color_Index"] * 50
    - selected_data["Disease_Score"] * 10
    + selected_data["Soil_Moisture_%"] * 0.5
)

st.progress(int(min(max(health_score, 0), 100)))
st.write(f"Health Score: {round(health_score,2)} / 100")

# Growth Chart
st.markdown("## 📈 Plant Growth Trend")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Date"], df["Plant_Height_cm"], marker="o")
plt.xticks(rotation=90)
plt.grid(True)
st.pyplot(fig)

# Moisture Chart
st.markdown("## 💦 Soil Moisture Analysis")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df["Date"], df["Soil_Moisture_%"], marker="o")
plt.xticks(rotation=90)
plt.grid(True)
st.pyplot(fig2)

# Disease Prediction Model
st.markdown("## 🔬 Disease Prediction Engine")

features = [
    "Temp_C",
    "Humidity_%",
    "Rainfall_mm",
    "Soil_Moisture_%",
    "Leaf_Color_Index"
]

X = df[features]
y = df["Disease_Score"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

temp = st.number_input("Temperature", value=32)
humidity = st.number_input("Humidity", value=75)
rainfall = st.number_input("Rainfall", value=10)
moisture = st.number_input("Soil Moisture", value=40)
leaf = st.number_input("Leaf Color Index", value=0.80)

if st.button("Predict Disease Risk"):
    sample = [[temp, humidity, rainfall, moisture, leaf]]
    prediction = model.predict(sample)

    if prediction[0] == 0:
        st.success("Healthy Crop 🌱")
    elif prediction[0] == 1:
        st.info("Low Risk")
    elif prediction[0] == 2:
        st.warning("Medium Risk")
    else:
        st.error("High Disease Risk 🚨")

# Weekly Summary
st.markdown("## 📋 Weekly Report")

df["Date"] = pd.to_datetime(df["Date"])
df["Week"] = df["Date"].dt.isocalendar().week

weekly_report = df.groupby("Week").agg({
    "Water_Given_Liters": "sum",
    "Fertilizer_Given_kg": "sum",
    "Plant_Height_cm": "mean",
    "Disease_Score": "mean"
}).reset_index()

st.dataframe(weekly_report)

# Footer
st.markdown("---")
st.markdown("### 🚜 CaneGuardian AI | Next Gen Sugarcane Farming")