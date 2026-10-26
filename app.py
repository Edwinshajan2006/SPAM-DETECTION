
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SMS Shield | Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "spam_model.joblib"

# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1020, #111a2e);
        color: #f1f5f9;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px;
        border: 1px solid #334155;
        border-radius: 20px;
        background: rgba(30, 41, 59, 0.8);
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #e2e8f0;
        font-size: 2.5rem;
    }

    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    div[data-testid="stMetric"] {
        background: #172238;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 14px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# SESSION STATE CALLBACKS
# ==================================================

def clear_message():
    st.session_state["sms_input"] = ""
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_message", None)


def load_example(text):
    st.session_state["sms_input"] = text
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_message", None)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Trained model not found. Make sure spam_model.joblib "
        "is in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SMS Shield</h1>
        <p>Machine-learning powered SMS spam detection</p>
        <p>
            Analyze a message and classify it as Spam or Ham
            using Natural Language Processing and Linear SVM.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# MESSAGE ANALYZER
# ==================================================

left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.subheader("🔍 Message Analyzer")
    st.write("Paste an SMS message below to check it.")

    message = st.text_area(
        "SMS message",
        placeholder="Example: Congratulations! You won a free prize...",
        height=170,
        key="sms_input",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)

    with col1:
        analyze = st.button(
            "🔎 Analyze Message",
            type="primary",
            use_container_width=True,
        )

    with col2:
        st.button(
            "Clear",
            use_container_width=True,
            on_click=clear_message,
        )

    # Run prediction only when Analyze is clicked.
    if analyze:
        if not message.strip():
            st.warning("Please enter a message before analyzing.")
            st.session_state.pop("last_prediction", None)
            st.session_state.pop("last_message", None)
        else:
            prediction = model.predict([message.strip()])[0]

            st.session_state["last_prediction"] = prediction
            st.session_state["last_message"] = message.strip()

    # Display the most recent prediction.
    if "last_prediction" in st.session_state:
        prediction = st.session_state["last_prediction"]

        if prediction == "spam":
            st.error("🚨 SPAM DETECTED")
            st.write(
                "The model classified this message as potential spam."
            )
        else:
            st.success("✅ HAM · LIKELY LEGITIMATE")
            st.write(
                "The model classified this message as likely legitimate."
            )

        st.caption(
            "Machine-learning predictions are not guaranteed. "
            "Verify suspicious messages independently."
        )

with right:
    st.subheader("🧪 Quick Examples")
    st.write("Load a sample message into the analyzer.")

    st.button(
        "Example: Spam",
        use_container_width=True,
        on_click=load_example,
        args=(
            "URGENT! Congratulations, you have won a free cash prize. "
            "Call now to claim your reward!",
        ),
    )

    st.button(
        "Example: Ham",
        use_container_width=True,
        on_click=load_example,
        args=(
            "Hey, are we meeting at the library after class today?",
        ),
    )

    st.info(
        "TF-IDF converts text into numerical features. "
        "The Linear SVM classifier uses those features to predict "
        "whether a message is spam or ham."
    )

st.divider()

# ==================================================
# MODEL PERFORMANCE
# ==================================================

st.subheader("📊 Model Performance")

st.caption(
    "Results from the held-out test set used during model evaluation."
)

metrics = [
    ("Accuracy", 98.45),
    ("Spam Precision", 93.08),
    ("Spam Recall", 94.53),
    ("Spam F1-score", 93.80),
]

metric_columns = st.columns(4)

for column, (name, value) in zip(metric_columns, metrics):
    column.metric(name, f"{value:.2f}%")

st.divider()

# ==================================================
# CONFUSION MATRIX
# ==================================================

st.subheader("Confusion Matrix")

st.write(
    "Actual labels compared with the predictions on the test dataset."
)

cm = pd.DataFrame(
    [[895, 9], [7, 121]],
    index=["Actual Ham", "Actual Spam"],
    columns=["Predicted Ham", "Predicted Spam"],
)

fig, ax = plt.subplots(figsize=(8, 4))

fig.patch.set_facecolor("#172238")
ax.set_facecolor("#172238")

ax.imshow(cm.values, cmap="Blues")

ax.set_xticks([0, 1], labels=cm.columns)
ax.set_yticks([0, 1], labels=cm.index)

ax.tick_params(colors="white", labelsize=10)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm.iloc[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=15,
            fontweight="bold",
        )

ax.set_xlabel("Predicted Label", color="white")
ax.set_ylabel("Actual Label", color="white")
ax.set_title("SMS Classification Results", color="white")

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "SMS Shield | Machine Learning Microproject | "
    "Python · scikit-learn · Streamlit"
)
