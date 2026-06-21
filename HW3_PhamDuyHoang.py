import os
import numpy as np
import pandas as pd
import streamlit as st
import base64
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


def load_data(base_path="data"):
    train_path = os.path.join(base_path, "raw_train.csv")
    val_path   = os.path.join(base_path, "raw_val.csv")
    test_path  = os.path.join(base_path, "raw_test.csv")

    df_train = pd.read_csv(train_path).dropna()
    df_val   = pd.read_csv(val_path).dropna()
    df_test  = pd.read_csv(test_path).dropna()

    feature_cols = [col for col in df_train.columns if col != 'target']

    X_train = df_train[feature_cols];  y_train = df_train['target']
    X_val   = df_val[feature_cols];    y_val   = df_val['target']
    X_test  = df_test[feature_cols];   y_test  = df_test['target']

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_label_distribution(y):
    counts      = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    dist = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
    dist.index = dist.index.map({0: 'Healthy (0)', 1: 'Heart Disease (1)'})
    return dist


def compute_feature_importance(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    return importances.sort_values(ascending=False)


def preprocess_raw_input(raw_dict):
    return {k: v for k, v in raw_dict.items() if v is not None}


def feature_selection(data_dict):
    feature_order = [
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal'
    ]
    return {feat: data_dict[feat] for feat in feature_order}


def feature_encoding(data_dict):
    enc = data_dict.copy()
    enc['sex']     = float(data_dict['sex'])
    enc['cp']      = float(data_dict['cp']    - 1) / 3.0
    enc['fbs']     = float(data_dict['fbs'])
    enc['restecg'] = float(data_dict['restecg']) / 2.0
    enc['exang']   = float(data_dict['exang'])
    enc['slope']   = float(data_dict['slope'] - 1) / 2.0
    enc['ca']      = float(data_dict['ca'])   / 3.0
    enc['thal']    = float(data_dict['thal']  - 3) / 4.0
    return enc


def feature_standardization(data_dict, scaler):
    df = pd.DataFrame([data_dict])
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[continuous_cols] = scaler.transform(df[continuous_cols])

    feature_order = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
        'sex', 'cp', 'fbs', 'restecg', 'exang',
        'slope', 'ca', 'thal'
    ]
    return df[feature_order]


def scale_raw_input(raw_dict):
    base_path = "data"
    if not os.path.exists(os.path.join(base_path, "raw_train.csv")):
        base_path = "."
        if not os.path.exists(os.path.join(base_path, "raw_train.csv")):
            base_path = os.path.dirname(os.path.abspath(__file__))
    
    X_train, _, _, _, _, _ = load_data(base_path=base_path)
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    computed_means = X_train[continuous_cols].mean().to_dict()
    computed_stds = X_train[continuous_cols].std().to_dict()
    
    # Check if the training data is already standardized (mean near 0, std near 1)
    is_standardized = abs(computed_means['age']) < 0.1 and abs(computed_stds['age'] - 1.0) < 0.1

    scaler = StandardScaler()
    if is_standardized:
        scaler.mean_ = np.array([54.549587, 130.958678, 249.838843, 149.962810, 0.999174])
        scaler.scale_ = np.array([8.978373, 17.586103, 52.737566, 22.639527, 1.120618])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_samples_seen_ = len(X_train)
        scaler.feature_names_in_ = np.array(continuous_cols)
    else:
        scaler.fit(X_train[continuous_cols])

    preprocessed = preprocess_raw_input(raw_dict)
    selected     = feature_selection(preprocessed)
    encoded      = feature_encoding(selected)
    standardized = feature_standardization(encoded, scaler)
    return standardized

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, selected_features):
    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]
    X_test_sel  = X_test[selected_features]

    dt  = DecisionTreeClassifier(max_depth=None, min_samples_leaf=15, criterion='entropy', random_state=0)
    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='chebyshev')
    nb  = GaussianNB(var_smoothing=1e-12)
    rf  = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=1, random_state=0)
    
    try:
        ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=0, algorithm="SAMME")
    except TypeError:
        ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=0)
        
    gb  = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.2, random_state=0)
    xgb = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.2, random_state=0, eval_metric='logloss')

    estimators = [
        ('dt',  dt),  ('knn', knn), ('nb',  nb),
        ('rf',  rf),  ('ada', ada), ('gb',  gb), ('xgb', xgb)
    ]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')

    models = {
        'Decision Tree':          dt,
        # 'k-NN':                   knn,
        # 'Naive Bayes':            nb,
        'AdaBoost':               ada,
        'Random Forest':          rf,
        'Gradient Boosting':      gb,
        'XGBoost':                xgb,
        # 'Ensemble (Soft Voting)': ensemble
    }

    results = {}

    for name, model in models.items():
        # Train
        model.fit(X_train_sel, y_train)

        y_val_pred = model.predict(X_val_sel)
        y_val_prob = (
            model.predict_proba(X_val_sel)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        val_acc  = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred, zero_division=0)
        val_rec  = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1   = f1_score(y_val, y_val_pred, zero_division=0)
        val_auc  = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else 0.5
        val_cm   = confusion_matrix(y_val, y_val_pred)

        y_test_pred = model.predict(X_test_sel)
        y_test_prob = (
            model.predict_proba(X_test_sel)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        test_acc  = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec  = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1   = f1_score(y_test, y_test_pred, zero_division=0)
        test_auc  = roc_auc_score(y_test, y_test_prob) if y_test_prob is not None else 0.5
        test_cm   = confusion_matrix(y_test, y_test_pred)

        results[name] = {
            'model': model,
            'val_metrics': {
                'Accuracy': val_acc, 'Precision': val_prec,
                'Recall': val_rec,   'F1-Score': val_f1,
                'ROC-AUC': val_auc,  'Confusion Matrix': val_cm
            },
            'test_metrics': {
                'Accuracy': test_acc, 'Precision': test_prec,
                'Recall': test_rec,   'F1-Score': test_f1,
                'ROC-AUC': test_auc,  'Confusion Matrix': test_cm
            }
        }

    return results

#sample
def get_example_patients():
    """Returns realistic example patients with raw clinical values."""
    return {
        "Example 1 (No Heart Disease)": {
            'age': 58, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 3
        },
        "Example 2 (Heart Disease)": {
            'age': 67, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 286,
            'fbs': 0, 'restecg': 0, 'thalach': 108, 'exang': 1,
            'oldpeak': 1.5, 'slope': 2, 'ca': 3, 'thal': 3
        }
    }


st.markdown("""
<style>
    /* Page background */
    .stApp {
        background-color: rgb(14, 17, 23);;
        color: white !important;
    }
    
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Expander Container */
    div[data-testid="stExpander"] {
        background-color: #262626 !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #262626 !important;
        color: white !important;
        font-weight: bold !important;
    }
    div[data-testid="stExpander"] div[role="region"] {
        background-color: #262626 !important;
    }
    div[data-testid="stExpander"] label {
        color: #DDDDDD !important;
        font-weight: 500 !important;
    }
    
    /* Dropdown Selection & Inputs */
    div[data-baseweb="select"] > div {
        background-color: #333333 !important;
        color: white !important;
        border: 1px solid #444444 !important;
    }
    div[data-baseweb="select"] svg {
        fill: white !important;
    }
    div[data-baseweb="select"] span {
        color: white !important;
    }
    input {
        background-color: #333333 !important;
        color: white !important;
        border: 1px solid #444444 !important;
    }
            
    .modebar-container > div{
        display: none;
    }
    
    /* Button Styling */
    button {
        background-color: #444444 !important;
        color: white !important;
        border: 1px solid #555555 !important;
        border-radius: 4px !important;
        padding: 6px 12px !important;
        font-weight: 500 !important;
    }
    button:hover {
        background-color: #555555 !important;
        color: white !important;
        border: 1px solid #666666 !important;
    }
    .stExpander > details > summary > span{
        flex-direction: row-reverse;
    }
    div[data-testid="stExpanderDetails"] > div > div{
        padding: 10px;
        border: solid 1px grey;
        border-radius: 10px;
    }
    .stHorizontalBlock > .stColumn:nth-child(2) > .stVerticalBlock{
        position: relative;        
    }
    .stHorizontalBlock > .stColumn:nth-child(2) > .stVerticalBlock > div:nth-child(2) {
        position: absolute;
        top: 50px;
        z-index: 999;        
    }
    div[data-testid="stNumberInputContainer"] > div > button{
        display: none;        
    } 
    .stVerticalBlock.st-emotion-cache-wfksaw.e1rw0b1u3 {
        top: 0;        
    }
</style>
""", unsafe_allow_html=True)

# 1. Load data and setup session states
X_train, y_train, X_val, y_val, X_test, y_test = load_data()
examples = get_example_patients()

# Pre-populate session state with Example 1 if empty
if 'age' not in st.session_state:
    for k, v in examples["Example 1 (No Heart Disease)"].items():
        st.session_state[k] = v


def load_example():
    """Callback when example selection changes."""
    sel = st.session_state.example_select
    if sel in examples:
        for k, v in examples[sel].items():
            st.session_state[k] = v


# Compute global feature importances using Random Forest once
@st.cache_resource
def get_global_feature_importance():
    return compute_feature_importance(X_train, y_train)


feature_importance = get_global_feature_importance()


# Cache model training on 13 features
@st.cache_resource
def get_cached_models(k_features):
    selected_features = list(feature_importance.index[:k_features])
    results = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, X_test, y_test, selected_features
    )
    return results, selected_features


# Set hyperparameter features to match all 13 clinical features
results, selected_features = get_cached_models(k_features=13)

# Build Layout columns
left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    # 13 Clinical inputs inside the expander container
    with st.expander("✍️ Enter Patient Features", expanded=True):
        # Row 1
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            st.number_input("age (years)", min_value=1, max_value=120, step=1, key="age")
        with r1_c2:
            st.selectbox(
                "sex (0=female, 1=male)", 
                options=[0, 1], 
                format_func=lambda x: "1" if x==1 else "0",
                key="sex"
            )
        with r1_c3:
            st.selectbox("cp (chest pain type 1..4)", options=[1, 2, 3, 4], key="cp")
        with r1_c4:
            st.number_input("trestbps (resting BP mmHg)", min_value=50, max_value=250, step=1, key="trestbps")
        
        # Row 2
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            st.number_input("chol (serum cholesterol mg/dl)", min_value=50, max_value=600, step=1, key="chol")
        with r2_c2:
            st.selectbox("fbs (>120 mg/dl? 1/0)", options=[0, 1], key="fbs")
        with r2_c3:
            st.selectbox("restecg (0..2)", options=[0, 1, 2], key="restecg")
        with r2_c4:
            st.number_input("thalach (max heart rate)", min_value=50, max_value=250, step=1, key="thalach")
        
        # Row 3
        r3_c1, r3_c2, r3_c3, r3_c4 = st.columns(4)
        with r3_c1:
            st.selectbox("exang (exercise angina 1/0)", options=[0, 1], key="exang")
        with r3_c2:
            st.number_input("oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1, key="oldpeak")
        with r3_c3:
            st.selectbox("slope (1..3)", options=[1, 2, 3], key="slope")
        with r3_c4:
            st.selectbox("ca (major vessels 0..3)", options=[0, 1, 2, 3], key="ca")
        
        # Row 4 (Full width)
        st.selectbox("thal (3=normal, 6=fixed, 7=reversible)", options=[3, 6, 7], key="thal")
        
        # Example Patient selector and Predict trigger
        r5_c1, r5_c2 = st.columns([2, 1])
        with r5_c1:
            st.selectbox(
                "Select Example Patient", 
                options=list(examples.keys()), 
                key="example_select", 
                on_change=load_example
            )
        with r5_c2:
            # st.markdown("<div style='height: 28px; top: 0;'></div>", unsafe_allow_html=True)
            predict_clicked = st.button("🔍 Predict", width='content')
            
# --- Right Column: Predictions Display ---
with right_col:
    # Prepare patient input dictionary
    raw_patient = {
        'age': st.session_state.age,
        'sex': st.session_state.sex,
        'cp': st.session_state.cp,
        'trestbps': st.session_state.trestbps,
        'chol': st.session_state.chol,
        'fbs': st.session_state.fbs,
        'restecg': st.session_state.restecg,
        'thalach': st.session_state.thalach,
        'exang': st.session_state.exang,
        'oldpeak': st.session_state.oldpeak,
        'slope': st.session_state.slope,
        'ca': st.session_state.ca,
        'thal': st.session_state.thal
    }
    
    # Scale patient input dynamically
    patient_scaled_df = scale_raw_input(raw_patient)
    
    # Run inference across models
    predictions = {}
    for name, info in results.items():
        model = info['model']
        patient_input = patient_scaled_df[selected_features]
        pred = model.predict(patient_input)[0]
        prob_1 = model.predict_proba(patient_input)[0, 1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
        
        predictions[name] = {
            'class': pred,
            'prob_1': prob_1
        }
        
    # Draw Plotly Bar Chart matching prototype
    models_order = ['Decision Tree', 'AdaBoost', 'Random Forest', 'Gradient Boosting', 'XGBoost']
    
    x_bars = []
    y_bars = []
    colors = []
    texts = []
    hovers = []
    
    for name in models_order:
        pred_info = predictions[name]
        p1 = pred_info['prob_1']
        c = pred_info['class']
        
        if c == 1:
            conf = p1
            color = '#C5314B'  # Crimson-Pink Red
            text = '🫀 Heart Disease'
            hover = f"<b>{name}</b><br>Prediction: Heart Disease<br>Confidence: {conf:.1%}"
        else:
            conf = 1.0 - p1
            color = '#2D7D31'  # Grass Green
            text = '✅ No Heart Disease'
            hover = f"<b>{name}</b><br>Prediction: No Heart Disease<br>Confidence: {conf:.1%}"
            
        x_bars.append(name)
        y_bars.append(conf)
        colors.append(color)
        texts.append(text)
        hovers.append(hover)
        
    fig_pred = go.Figure()

    # Draw 7 base classifiers (no border)
    fig_pred.add_trace(go.Bar(
        x=x_bars[:-1],
        y=y_bars[:-1],
        text=texts[:-1],
        textposition='inside',
        insidetextanchor='end',
        marker=dict(
            color=colors[:-1],
            line=dict(width=0)
        ),
        hoverinfo='text',
        hovertext=hovers[:-1],
        textfont=dict(color='white', size=11, family='sans-serif'),
        textangle=90,
        showlegend=False
    ))

    # Draw Ensemble Classifier (with border)
    fig_pred.add_trace(go.Bar(
        x=[x_bars[-1]],
        y=[y_bars[-1]],
        text=[texts[-1]],
        textposition='inside',
        insidetextanchor='end',
        marker=dict(
            color=[colors[-1]],
            line=dict(color='black', width=2.5)
        ),
        hoverinfo='text',
        hovertext=[hovers[-1]],
        textfont=dict(color='white', size=11, family='sans-serif'),
        textangle=90,
        showlegend=False
    ))
    
    # Add text labels above bars
    annotations = []
    for idx, (name, val) in enumerate(zip(x_bars, y_bars)):
        annotations.append(dict(
            x=name,
            y=val + 0.01,
            text=f"<b>{int(round(val*100))}%</b>",
            showarrow=False,
            font=dict(color='black', size=12, family='sans-serif'),
            xanchor='center',
            yanchor='bottom'
        ))
        
    fig_pred.update_layout(
        title={
            'text': "Model Predictions",
            'y': 0.95,
            'x': 0.05,
            'xanchor': 'left',
            'yanchor': 'top',   
            'font': dict(size=18, color='black')
        },
        yaxis=dict(
            title=dict(text="Prediction Confidence", font=dict(color='black', size=13)),
            range=[0, 1.15],
            showgrid=False,
            showline=False,
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"],
            gridcolor='#E5E7EB',
            linecolor='black',
            linewidth=1.5,
            tickcolor='black',
            tickfont=dict(color='black', size=15)
        ),
        xaxis=dict(
            title=dict(text="Model", font=dict(color='black', size=13)),
            linecolor='black',
            linewidth=1.5,
            tickcolor='black',
            tickfont=dict(color='black', size=15),
            tickangle=35,
            showgrid=False,
            showline=False,
        ),
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=50, r=20, t=60, b=80),
        showlegend=False,
        annotations=annotations
    )

    with open("assets/plot.png", "rb") as f:
        img = base64.b64encode(f.read()).decode()

    st.markdown(f"""
<div style="background-color: white; border: 1px solid #444444; border-radius: 10px; height: 30em; color: black; font-family: sans-serif;">
  <div style="background-color: #2A2F35; color: white; border: 2px solid black; border-radius: 10px 0 10px 0; padding: 5px 12px; font-size: 13px; font-weight: bold; display: inline-block; margin-bottom: 8px;">
    <img src="data:image/png;base64,{img}" width="24"><span>Model Predictions Overview</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.plotly_chart(fig_pred, width='content')

st.set_page_config(
    page_title="Homework 3 Pham Duy Hoang",
    page_icon="heart",
    layout="wide",
)