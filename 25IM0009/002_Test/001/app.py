import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import copy

st.set_page_config(page_title="Privacy-Preserving Fraud Detection", layout="wide")

# Initialize session state logs and model updates
if "logs" not in st.session_state:
    st.session_state.logs = []

if "updates" not in st.session_state:
    st.session_state.updates = []

if "global_model" not in st.session_state:
    st.session_state.global_model = None

if "version" not in st.session_state:
    st.session_state.version = 1

# Generate datasets for two banks
X1, y1 = make_classification(n_samples=800, n_features=10, weights=[0.9, 0.1], random_state=42)
X2, y2 = make_classification(n_samples=800, n_features=10, weights=[0.85, 0.15], random_state=7)

st.title("🏦 Privacy-Preserving Cross-Bank Fraud Detection Demo")
role = st.selectbox("Select Your Role:", ["Bank A Admin", "Bank B Admin", "Central Admin", "Auditor"])

def bank_dashboard(name, X, y):
    st.subheader(f"{name} Dashboard")
    model = LogisticRegression()

    if st.button("Train Local Model"):
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        st.session_state[f"{name}_model"] = model
        st.success(f"✅ Local training done. Accuracy: {acc:.2f}")
        st.session_state.logs.append(f"{name} trained local model.")

    if st.button("Send Model Update"):
        if f"{name}_model" in st.session_state:
            m = st.session_state[f"{name}_model"]
            st.session_state.updates.append(
                (copy.deepcopy(m.coef_), copy.deepcopy(m.intercept_))
            )
            st.success("✅ Model update sent (only weights, no raw data).")
            st.session_state.logs.append(f"{name} sent model update.")
        else:
            st.warning("⚠ Please train the local model first.")

def central_admin_dashboard():
    st.subheader("Central Aggregator Dashboard")

    if st.button("Aggregate Model Updates"):
        if len(st.session_state.updates) < 2:
            st.warning("⚠ Need at least 2 updates to aggregate.")
            return
        avg_coef = np.mean([u[0] for u in st.session_state.updates], axis=0)
        avg_intercept = np.mean([u[1] for u in st.session_state.updates], axis=0)

        global_model = LogisticRegression()
        global_model.coef_ = avg_coef
        global_model.intercept_ = avg_intercept
        global_model.classes_ = np.array([0, 1])

        st.session_state.global_model = global_model
        st.session_state.version += 1
        st.success(f"✅ Global Model v{st.session_state.version} created.")
        st.session_state.logs.append("Aggregator created global model.")

    if st.button("Approve Global Model"):
        if st.session_state.global_model is not None:
            st.success(f"✅ Global Model v{st.session_state.version} approved.")
            st.session_state.logs.append("Central Admin approved global model.")
        else:
            st.warning("⚠ No global model available to approve.")

def auditor_dashboard():
    st.subheader("Audit Logs")
    if len(st.session_state.logs) == 0:
        st.info("No logs available yet.")
    else:
        for log in st.session_state.logs:
            st.write(log)

# Role-based UI routing
if role == "Bank A Admin":
    bank_dashboard("Bank A", X1, y1)
elif role == "Bank B Admin":
    bank_dashboard("Bank B", X2, y2)
elif role == "Central Admin":
    central_admin_dashboard()
elif role == "Auditor":
    auditor_dashboard()

# Show global model performance if available
if st.session_state.global_model is not None:
    st.divider()
    st.subheader(f"Global Model v{st.session_state.version} Performance")
    accA = accuracy_score(y1, st.session_state.global_model.predict(X1))
    accB = accuracy_score(y2, st.session_state.global_model.predict(X2))
    st.write(f"Accuracy on Bank A Dataset: **{accA:.2f}**")
    st.write(f"Accuracy on Bank B Dataset: **{accB:.2f}**")
