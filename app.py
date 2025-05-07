import streamlit as st
import pandas as pd
from data_preprocesing import load_data, encode_labels, split_features_and_target, split_train_and_test_data
from model import train_model, predict, evaluate_model

st.title("Income Prediction App")

