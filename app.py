from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

st.title("Confusion Matrix Visualizer")

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
cm_disp = (
        ConfusionMatrixDisplay(cm, display_labels=display_labels)
        .plot(cmap=cmap)
    )
cm_disp.ax_.set_title(cm_title)
st.write(cm_disp.figure_)

st.subheader("Metrics")

@st.cache
def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> tuple:
    try:
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        accuracy = precision = recall = f1 = 0
    return (
        round(accuracy, 3),
        round(precision, 3),
        round(recall, 3),
        round(f1, 3)
    )

st.write("Accuracy: ", calculate_metrics(tp, fp, fn, tn)[0])
st.write("Precision: ", calculate_metrics(tp, fp, fn, tn)[1])
st.write("Recall: ", calculate_metrics(tp, fp, fn, tn)[2])
st.write("F1: ", calculate_metrics(tp, fp, fn, tn)[3])
