import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Sales AI", layout="wide")

# ---------------- BACKGROUND STYLE ----------------
def add_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1551288049-bebda4e38f71");
            background-size: cover;
        }
        .block-container {
            background: rgba(255,255,255,0.9);
            padding: 2rem;
            border-radius: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- LOGIN SYSTEM ----------------
def login():
    add_bg()
    st.title("🔐 Smart Sales AI Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- TOP BAR ----------------
col1, col2 = st.columns([8,1])
with col2:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

if st.button("⬅ Back to Login"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- LOAD MODEL ----------------
model, le_category, le_region, le_segment = pickle.load(open("model/model.pkl", "rb"))

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Superstore.csv")

# ---------------- DASHBOARD ----------------
st.title("📊 Smart Sales Prediction Dashboard")

c1, c2, c3 = st.columns(3)

with c1:
    category = st.selectbox("Select Category", le_category.classes_)

with c2:
    region = st.selectbox("Select Region", le_region.classes_)

with c3:
    segment = st.selectbox("Select Segment", le_segment.classes_)

# ---------------- PDF FUNCTION ----------------
def create_pdf(prediction, insight, recommendation):
    pdf_path = "Sales_Report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Smart Sales AI Report</b>", styles["Title"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Predicted Sales: ₹ {round(prediction,2)}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Insight: {insight}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Recommendation: {recommendation}", styles["Normal"]))

    doc.build(elements)
    return pdf_path

# ---------------- PREDICTION ----------------
if st.button("Predict Sales"):

    cat_encoded = le_category.transform([category])[0]
    reg_encoded = le_region.transform([region])[0]
    seg_encoded = le_segment.transform([segment])[0]

    prediction = model.predict([[cat_encoded, reg_encoded, seg_encoded]])[0]

    if prediction < 500:
        insight = "Sales are relatively low"
        recommendation = "Consider promotions or marketing strategies"
    else:
        insight = "Sales performance is strong"
        recommendation = "Maintain strategy and increase stock"

    st.success(f"Predicted Sales: ₹ {round(prediction,2)}")
    st.write("### 📌 Insight:", insight)
    st.write("### 💡 Recommendation:", recommendation)

    # ---------- SMALL GRAPHS ----------
    st.subheader("📊 Visual Analysis")

    colA, colB, colC = st.columns(3)

    # Bar graph
    fig1, ax1 = plt.subplots(figsize=(4,3))
    ax1.bar(["Predicted Sales"], [prediction])
    ax1.set_ylabel("Sales Amount")

    # Histogram
    fig2, ax2 = plt.subplots(figsize=(4,3))
    ax2.hist(data["Sales"], bins=30)
    ax2.set_xlabel("Sales")

    # Feature importance
    fig3, ax3 = plt.subplots(figsize=(4,3))
    importance = model.feature_importances_
    features = ["Category", "Region", "Segment"]
    ax3.bar(features, importance)

    with colA:
        st.pyplot(fig1)
    with colB:
        st.pyplot(fig2)
    with colC:
        st.pyplot(fig3)

    # PDF Download
    pdf_file = create_pdf(prediction, insight, recommendation)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download Sales Report",
            data=f,
            file_name="Sales_Report.pdf",
            mime="application/pdf"
        )