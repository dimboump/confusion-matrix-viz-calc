from __future__ import annotations

from typing import List
from typing import Optional

import streamlit as st
import numpy as np
from matplotlib.pyplot import rc_context
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

st.title("Confusion Matrix Generator")

st.sidebar.subheader("Enter the desired values:")
tp = st.sidebar.number_input("True Positives" , value=0, min_value=0)
fp = st.sidebar.number_input("False Positives", value=0, min_value=0)
fn = st.sidebar.number_input("False Negatives", value=0, min_value=0)
tn = st.sidebar.number_input("True Negatives" , value=0, min_value=0)

# Advanced options
with st.sidebar.expander("Show advanced options"):
    cm_title = st.text_input("Title", value="")
    cm_title = cm_title if cm_title else None
    display_labels = st.text_input("Display labels (comma separated)", value="")
    display_labels = display_labels.split(",") if display_labels else None
    cmap = st.selectbox("Colormap", ["Blues", "Greens", "Reds", "Purples", "Oranges", "Greys"])

# Main section
cm = np.array([[tp, fp], [fn, tn]])

def plot_cm(
    cm: np.ndarray, 
    *,
    title: Optional[str] = None, 
    display_labels: List[str] = None, 
    cmap: str = "Blues"
) -> None:
    if display_labels is None:
        display_labels = ["0", "1"]

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(cmap=cmap, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

with st.container():
    plot_cm(cm, title=cm_title, display_labels=display_labels, cmap=cmap)

st.subheader("Metrics")

@st.cache
def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    try:
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        accuracy = precision = recall = f1 = 0
    return {
        "Accuracy: ": accuracy,
        "Precision: ": precision,
        "Recall: ": recall,
        "F1: ": f1
    }

metrics = calculate_metrics(tp, fp, fn, tn)
col1, col2, col3, col4 = st.columns(4)
for col, metric, score in zip((col1, col2, col3, col4), metrics.keys(), metrics.values()):
    with col:
        st.metric(label=metric, value=round(score, 3))
