# ============================================================
#  CAR PRICE ANALYSIS — Streamlit Dashboard
#  Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Car Price Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #1f77b4;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 13px;
        color: #666;
        margin-top: 4px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .prediction-price {
        font-size: 48px;
        font-weight: bold;
    }
    .category-cheap     { background: #28a745; color: white; padding: 8px 20px; border-radius: 20px; font-size: 18px; font-weight: bold; }
    .category-moderate  { background: #fd7e14; color: white; padding: 8px 20px; border-radius: 20px; font-size: 18px; font-weight: bold; }
    .category-expensive { background: #dc3545; color: white; padding: 8px 20px; border-radius: 20px; font-size: 18px; font-weight: bold; }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  LOAD & TRAIN MODELS (cached so it only runs once)
# ============================================================
@st.cache_resource
def load_and_train():
    df = pd.read_csv("car_price.csv")

    numerical_cols   = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
    categorical_cols = ['model', 'transmission', 'fuelType', 'Make']

    # Fill nulls
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # IQR capping
    for col in ['price', 'mileage', 'engineSize']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Classification target
    low_thresh  = df['price'].quantile(0.33)
    high_thresh = df['price'].quantile(0.67)

    def price_category(p):
        if p <= low_thresh:   return 'Cheap'
        elif p <= high_thresh: return 'Moderate'
        else:                  return 'Expensive'

    df['price_category'] = df['price'].apply(price_category)

    # Split
    X = df.drop(columns=['price', 'price_category'])
    y_reg = df['price']
    y_clf = df['price_category']

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _,            y_train_clf, y_test_clf  = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # Target encoding
    global_mean = y_train_reg.mean()
    train_temp  = X_train.copy()
    train_temp['price'] = y_train_reg.values
    target_encoders = {}
    for col in categorical_cols:
        mapping = train_temp.groupby(col)['price'].mean()
        target_encoders[col] = mapping
        X_train[col] = X_train[col].map(mapping).fillna(global_mean)
        X_test[col]  = X_test[col].map(mapping).fillna(global_mean)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_reg)
    y_pred_reg = lr.predict(X_test_scaled)

    # Train KNN (best params from grid search)
    knn = KNeighborsClassifier(n_neighbors=9, metric='manhattan')
    knn.fit(X_train_scaled, y_train_clf)
    y_pred_clf = knn.predict(X_test_scaled)

    # Metrics
    metrics = {
        'mae' : mean_absolute_error(y_test_reg, y_pred_reg),
        'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
        'r2'  : r2_score(y_test_reg, y_pred_reg),
        'mape': np.mean(np.abs((y_test_reg - y_pred_reg) / y_test_reg)) * 100,
        'acc' : accuracy_score(y_test_clf, y_pred_clf),
        'f1'  : f1_score(y_test_clf, y_pred_clf, average='weighted'),
        'prec': precision_score(y_test_clf, y_pred_clf, average='weighted'),
        'rec' : recall_score(y_test_clf, y_pred_clf, average='weighted'),
    }

    return {
        'df': df,
        'lr': lr,
        'knn': knn,
        'scaler': scaler,
        'target_encoders': target_encoders,
        'global_mean': global_mean,
        'categorical_cols': categorical_cols,
        'X_test_scaled': X_test_scaled,
        'y_test_reg': y_test_reg,
        'y_test_clf': y_test_clf,
        'y_pred_reg': y_pred_reg,
        'y_pred_clf': y_pred_clf,
        'metrics': metrics,
        'low_thresh': low_thresh,
        'high_thresh': high_thresh,
        'X_train_scaled': X_train_scaled,
    }

# ============================================================
#  SIDEBAR NAVIGATION
# ============================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3774/3774278.png", width=80)
st.sidebar.title("🚗 Car Price Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA", "🔮 Predict Price", "📈 Model Performance", "⚙️ Sensitivity Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")

data = load_and_train()
df   = data['df']

st.sidebar.metric("Total Cars",    f"{len(df):,}")
st.sidebar.metric("Features",      "9")
st.sidebar.metric("Price Range",   f"£{df['price'].min():,.0f} – £{df['price'].max():,.0f}")
st.sidebar.markdown("---")
st.sidebar.markdown("*Machine Learning Assignment 1*")
st.sidebar.markdown("*Linear Regression + KNN*")

# ============================================================
#  PAGE 1: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("🚗 Car Price Analysis Dashboard")
    st.markdown("#### Machine Learning Assignment 1 — Linear Regression & KNN Classification")
    st.markdown("---")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cars",       f"{len(df):,}")
    with col2:
        st.metric("Avg Price",        f"£{df['price'].mean():,.0f}")
    with col3:
        st.metric("Regression R²",    f"{data['metrics']['r2']:.3f}")
    with col4:
        st.metric("KNN Accuracy",     f"{data['metrics']['acc']*100:.1f}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 📋 Dataset Summary")
        summary = df[['year','price','mileage','tax','mpg','engineSize']].describe().round(2)
        st.dataframe(summary, use_container_width=True)

    with col_r:
        st.markdown("### 🎯 Price Category Distribution")
        counts = df['price_category'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ['#28a745','#fd7e14','#dc3545']
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90,
               wedgeprops=dict(edgecolor='white', linewidth=2))
        ax.set_title("Price Categories", fontweight='bold')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("### 🗂️ Sample Data")
    st.dataframe(df.drop(columns=['price_category']).head(10), use_container_width=True)

# ============================================================
#  PAGE 2: EDA
# ============================================================
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "🏷️ Categories"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df['price'], bins=60, color='steelblue', edgecolor='white', alpha=0.85)
            ax.axvline(df['price'].median(), color='tomato',  linestyle='--', lw=2, label=f"Median £{df['price'].median():,.0f}")
            ax.axvline(df['price'].mean(),   color='orange',  linestyle='--', lw=2, label=f"Mean   £{df['price'].mean():,.0f}")
            ax.set_title("Price Distribution", fontweight='bold')
            ax.set_xlabel("Price (£)")
            ax.legend()
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df['mileage'], df['price'], alpha=0.15, s=5, color='darkorchid')
            ax.set_title("Mileage vs Price", fontweight='bold')
            ax.set_xlabel("Mileage"); ax.set_ylabel("Price (£)")
            st.pyplot(fig); plt.close()

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            order = df.groupby('Make')['price'].median().sort_values(ascending=False).index
            sns.boxplot(data=df, x='Make', y='price', order=order, palette='tab10', ax=ax)
            ax.set_title("Price by Make", fontweight='bold')
            ax.tick_params(axis='x', rotation=20)
            st.pyplot(fig); plt.close()

        with col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            order2 = df.groupby('fuelType')['price'].median().sort_values(ascending=False).index
            sns.boxplot(data=df, x='fuelType', y='price', order=order2, palette='Set2', ax=ax)
            ax.set_title("Price by Fuel Type", fontweight='bold')
            ax.tick_params(axis='x', rotation=15)
            st.pyplot(fig); plt.close()

    with tab2:
        num_cols = ['year','price','mileage','tax','mpg','engineSize']
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, linewidths=0.5, ax=ax, square=True)
        ax.set_title("Correlation Heatmap", fontweight='bold', fontsize=13)
        st.pyplot(fig); plt.close()

        st.markdown("#### Top correlations with Price:")
        corr_price = corr['price'].drop('price').sort_values(key=abs, ascending=False)
        for feat, val in corr_price.items():
            bar_color = "🟥" if val > 0.3 else ("🟦" if val < -0.3 else "⬜")
            st.markdown(f"{bar_color} **{feat}**: `{val:.4f}`")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            low_thresh  = data['low_thresh']
            high_thresh = data['high_thresh']
            ax.hist(df['price'], bins=60, color='steelblue', edgecolor='white', alpha=0.7)
            ax.axvline(low_thresh,  color='green',  linestyle='--', lw=2, label=f'£{low_thresh:,.0f} (Cheap/Moderate)')
            ax.axvline(high_thresh, color='red',    linestyle='--', lw=2, label=f'£{high_thresh:,.0f} (Moderate/Expensive)')
            ax.set_title("Price Thresholds", fontweight='bold')
            ax.set_xlabel("Price (£)"); ax.legend()
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            order = ['Cheap', 'Moderate', 'Expensive']
            palette = {'Cheap':'#28a745','Moderate':'#fd7e14','Expensive':'#dc3545'}
            sns.violinplot(data=df, x='price_category', y='price',
                           order=order, palette=palette, ax=ax, inner='box')
            ax.set_title("Price Distribution per Category", fontweight='bold')
            st.pyplot(fig); plt.close()

# ============================================================
#  PAGE 3: PREDICT PRICE
# ============================================================
elif page == "🔮 Predict Price":
    st.title("🔮 Car Price Predictor")
    st.markdown("Fill in the car details and get an instant price prediction!")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown("### 🚘 Car Details")

        make = st.selectbox("Car Make", sorted(df['Make'].unique()))
        models_for_make = sorted(df[df['Make'] == make]['model'].unique())
        model_name = st.selectbox("Model", models_for_make)

        col_a, col_b = st.columns(2)
        with col_a:
            year        = st.slider("Year",         2000, 2024, 2018)
            mileage     = st.number_input("Mileage (miles)", 0, 200000, 20000, step=1000)
            engine_size = st.selectbox("Engine Size (L)", [1.0,1.2,1.4,1.5,1.6,1.8,2.0,2.2,2.5,3.0,4.0])
        with col_b:
            transmission = st.selectbox("Transmission", df['transmission'].unique())
            fuel_type    = st.selectbox("Fuel Type",    df['fuelType'].unique())
            tax          = st.number_input("Road Tax (£)", 0, 600, 145, step=5)
            mpg          = st.number_input("MPG", 10.0, 120.0, 55.0, step=0.5)

        predict_btn = st.button("🔮 Predict Price", use_container_width=True, type="primary")

    with col_out:
        st.markdown("### 📊 Prediction Result")

        if predict_btn:
            # Build input row
            input_df = pd.DataFrame([{
                'model'       : model_name,
                'year'        : float(year),
                'transmission': transmission,
                'mileage'     : float(mileage),
                'fuelType'    : fuel_type,
                'tax'         : float(tax),
                'mpg'         : float(mpg),
                'engineSize'  : float(engine_size),
                'Make'        : make
            }])

            # Target encode
            for col in data['categorical_cols']:
                input_df[col] = input_df[col].map(
                    data['target_encoders'][col]
                ).fillna(data['global_mean'])

            # Scale
            input_scaled = pd.DataFrame(
                data['scaler'].transform(input_df),
                columns=input_df.columns
            )

            # Predict
            price_pred    = data['lr'].predict(input_scaled)[0]
            category_pred = data['knn'].predict(input_scaled)[0]

            # Clamp to valid range
            price_pred = max(500, min(price_pred, 35000))

            # Category styling
            cat_colors = {'Cheap':'#28a745','Moderate':'#fd7e14','Expensive':'#dc3545'}
            cat_emoji  = {'Cheap':'💚','Moderate':'🟠','Expensive':'🔴'}
            color = cat_colors[category_pred]

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                        border-radius:15px;padding:30px;text-align:center;color:white;margin:10px 0">
                <div style="font-size:16px;opacity:0.85;margin-bottom:8px">Estimated Price</div>
                <div style="font-size:52px;font-weight:bold">£{price_pred:,.0f}</div>
                <div style="margin-top:16px">
                    <span style="background:{color};padding:8px 24px;border-radius:20px;
                                 font-size:18px;font-weight:bold">
                        {cat_emoji[category_pred]} {category_pred}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### What this means:")
            low_t  = data['low_thresh']
            high_t = data['high_thresh']
            if category_pred == 'Cheap':
                st.info(f"This car is in the **Cheap** category (≤ £{low_t:,.0f}). Good value for money!")
            elif category_pred == 'Moderate':
                st.warning(f"This car is in the **Moderate** category (£{low_t:,.0f} – £{high_t:,.0f}). Mid-range pricing.")
            else:
                st.error(f"This car is in the **Expensive** category (> £{high_t:,.0f}). Premium pricing.")

            # Show similar cars from dataset
            st.markdown("##### 🔍 Similar cars in dataset:")
            similar = df[
                (df['Make'] == make) &
                (df['price'].between(price_pred * 0.85, price_pred * 1.15))
            ][['model','year','mileage','price']].head(5)
            if len(similar) > 0:
                st.dataframe(similar, use_container_width=True)
            else:
                st.caption("No exact matches found in range.")
        else:
            st.markdown("""
            <div style="background:#f8f9fa;border-radius:10px;padding:40px;
                        text-align:center;color:#999;border:2px dashed #ddd">
                <div style="font-size:48px">🚗</div>
                <div style="font-size:16px;margin-top:10px">
                    Fill in the car details and click<br><b>Predict Price</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
#  PAGE 4: MODEL PERFORMANCE
# ============================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📉 Linear Regression", "🎯 KNN Classification"])

    with tab1:
        m = data['metrics']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²",   f"{m['r2']:.4f}",   help="Proportion of variance explained")
        col2.metric("MAE",  f"£{m['mae']:,.0f}", help="Average absolute error")
        col3.metric("RMSE", f"£{m['rmse']:,.0f}",help="Root mean squared error")
        col4.metric("MAPE", f"{m['mape']:.1f}%", help="Mean absolute percentage error")

        col_l, col_r = st.columns(2)
        with col_l:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(data['y_test_reg'], data['y_pred_reg'],
                       alpha=0.2, s=8, color='steelblue')
            mn = data['y_test_reg'].min(); mx = data['y_test_reg'].max()
            ax.plot([mn,mx],[mn,mx],'r--',lw=2,label='Perfect fit')
            ax.set_title("Predicted vs Actual", fontweight='bold')
            ax.set_xlabel("Actual Price (£)")
            ax.set_ylabel("Predicted Price (£)")
            ax.text(0.05, 0.92, f"R² = {m['r2']:.3f}",
                    transform=ax.transAxes, fontsize=11,
                    color='darkred', fontweight='bold')
            ax.legend()
            st.pyplot(fig); plt.close()

        with col_r:
            residuals = data['y_test_reg'].values - data['y_pred_reg']
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(residuals, bins=60, color='darkorchid', edgecolor='white', alpha=0.8)
            ax.axvline(0, color='red', linestyle='--', lw=2)
            ax.set_title("Residual Distribution", fontweight='bold')
            ax.set_xlabel("Residual (£)")
            st.pyplot(fig); plt.close()

        # Feature coefficients
        st.markdown("#### Feature Coefficients")
        coef_df = pd.DataFrame({
            'Feature'    : data['X_train_scaled'].columns,
            'Coefficient': data['lr'].coef_
        }).sort_values('Coefficient', key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['tomato' if c < 0 else 'steelblue' for c in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'],
                color=colors, edgecolor='white', alpha=0.85)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title("Feature Coefficients (scaled)", fontweight='bold')
        st.pyplot(fig); plt.close()

    with tab2:
        m = data['metrics']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{m['acc']*100:.2f}%")
        col2.metric("Precision", f"{m['prec']*100:.2f}%")
        col3.metric("Recall",    f"{m['rec']*100:.2f}%")
        col4.metric("F1-Score",  f"{m['f1']*100:.2f}%")

        col_l, col_r = st.columns(2)
        labels = ['Cheap', 'Moderate', 'Expensive']
        cm = confusion_matrix(data['y_test_clf'], data['y_pred_clf'], labels=labels)

        with col_l:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels,
                        linewidths=0.5, ax=ax)
            ax.set_title("Confusion Matrix (counts)", fontweight='bold')
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            st.pyplot(fig); plt.close()

        with col_r:
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                        xticklabels=labels, yticklabels=labels,
                        linewidths=0.5, ax=ax)
            ax.set_title("Confusion Matrix (normalized)", fontweight='bold')
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            st.pyplot(fig); plt.close()

        st.markdown("#### Classification Report")
        report = classification_report(data['y_test_clf'], data['y_pred_clf'],
                                        target_names=labels, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

# ============================================================
#  PAGE 5: SENSITIVITY ANALYSIS
# ============================================================
elif page == "⚙️ Sensitivity Analysis":
    st.title("⚙️ Sensitivity Analysis")
    st.markdown("---")

    m = data['metrics']

    st.markdown("### A) Impact of Removing Most Correlated Feature (engineSize)")
    from sklearn.model_selection import train_test_split as tts

    df2 = data['df'].copy()
    for col in ['model', 'transmission', 'fuelType', 'Make']:
        df2[col] = df2[col].map(data['target_encoders'][col]).fillna(data['global_mean'])

    X2  = df2.drop(columns=['price', 'price_category'])
    y2  = df2['price']
    Xtr, Xte, ytr, yte = tts(X2, y2, test_size=0.2, random_state=42)

    sc2   = StandardScaler()
    Xtr_s = pd.DataFrame(sc2.fit_transform(Xtr), columns=X2.columns)
    Xte_s = pd.DataFrame(sc2.transform(Xte),     columns=X2.columns)

    lr_full = LinearRegression().fit(Xtr_s, ytr)
    r2_full = r2_score(yte, lr_full.predict(Xte_s))

    lr_noe  = LinearRegression().fit(Xtr_s.drop(columns=['engineSize']), ytr)
    r2_noe  = r2_score(yte, lr_noe.predict(Xte_s.drop(columns=['engineSize'])))

    col1, col2, col3 = st.columns(3)
    col1.metric("R² with engineSize",    f"{r2_full:.4f}")
    col2.metric("R² without engineSize", f"{r2_noe:.4f}")
    col3.metric("Drop in R²",            f"{r2_full - r2_noe:.4f}",
                delta=f"-{r2_full-r2_noe:.4f}", delta_color="inverse")

    st.info("engineSize has a moderate impact on regression. The model degrades but still performs reasonably — this shows the other features carry meaningful signal too.")

    st.markdown("---")
    st.markdown("### B) KNN Without Scaling")

    df3 = data['df'].copy()
    for col in ['model','transmission','fuelType','Make']:
        df3[col] = df3[col].map(data['target_encoders'][col]).fillna(data['global_mean'])
    X3  = df3.drop(columns=['price','price_category'])
    y3  = df3['price_category']
    X3tr, X3te, y3tr, y3te = tts(X3, y3, test_size=0.2, random_state=42)

    knn_noscale = KNeighborsClassifier(n_neighbors=9, metric='manhattan')
    knn_noscale.fit(X3tr, y3tr)
    acc_noscale = accuracy_score(y3te, knn_noscale.predict(X3te))

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy WITH scaling",    f"{m['acc']*100:.2f}%")
    col2.metric("Accuracy WITHOUT scaling", f"{acc_noscale*100:.2f}%")
    col3.metric("Accuracy Drop",
                f"{(m['acc']-acc_noscale)*100:.2f}%",
                delta=f"-{(m['acc']-acc_noscale)*100:.2f}%",
                delta_color="inverse")

    st.warning("Scaling is critical for KNN! Without it, features with large ranges (mileage: 0–70k) dominate features with small ranges (engineSize: 1–3), completely distorting the distance calculations.")

    st.markdown("---")
    st.markdown("### C) Different Price Category Thresholds")

    alt_low  = st.slider("Alternative Low Threshold (£)",  5000, 15000, 10000, 500)
    alt_high = st.slider("Alternative High Threshold (£)", 15000, 30000, 20000, 500)

    if alt_low < alt_high:
        def cat_alt(p):
            if p <= alt_low:   return 'Cheap'
            elif p <= alt_high: return 'Moderate'
            else:               return 'Expensive'

        df3['price_category_alt'] = data['df']['price'].apply(cat_alt)
        _, _, y_alt_tr, y_alt_te = tts(X3, df3['price_category_alt'], test_size=0.2, random_state=42)

        Xtr_s2 = pd.DataFrame(StandardScaler().fit_transform(X3tr), columns=X3.columns)
        Xte_s2 = pd.DataFrame(StandardScaler().fit_transform(X3te), columns=X3.columns)

        knn_alt = KNeighborsClassifier(n_neighbors=9, metric='manhattan')
        knn_alt.fit(Xtr_s2, y_alt_tr)
        acc_alt = accuracy_score(y_alt_te, knn_alt.predict(Xte_s2))

        col1, col2 = st.columns(2)
        col1.metric("Original Thresholds Accuracy", f"{m['acc']*100:.2f}%")
        col2.metric("New Thresholds Accuracy",       f"{acc_alt*100:.2f}%")

        counts_alt = df3['price_category_alt'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#28a745','#fd7e14','#dc3545']
        ax.bar(counts_alt.index, counts_alt.values, color=colors, edgecolor='white', alpha=0.85)
        for i,(cat,val) in enumerate(counts_alt.items()):
            ax.text(i, val+200, f'{val:,}', ha='center', fontsize=10, fontweight='bold')
        ax.set_title("Class Distribution with New Thresholds", fontweight='bold')
        st.pyplot(fig); plt.close()
    else:
        st.error("Low threshold must be less than high threshold!")
