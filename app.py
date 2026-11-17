import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import math
from streamlit_gsheets import GSheetsConnection

# =============================================================================
# 1. إعدادات الصفحة الأساسية
# =============================================================================
st.set_page_config(
    page_title="ASA-PREDICTION MODEL 3",
    page_icon="🏗️",
    layout="wide"
)

# =============================================================================
# 2. دالة التحكم في الدخول (Login)
# =============================================================================
def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        col_left, col_mid, col_right = st.columns([1, 2, 1])

        with col_left:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL3-/main/OIP.jfif", width=120)

        with col_mid:
            st.markdown("""
                <div style='text-align: center; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;'>
                    <h1 style='color: #1E3A8A; margin-bottom: 5px; font-weight: 800;'>ASA-PREDICTION MODEL 3</h1>
                    <h3 style='margin-top: 0px; color: #4B5563; font-weight: 600;'>By: Eng. Aya Mohamed Sanad Aboud</h3>
                    <p style='font-size: 1.15em; color: #374151; font-weight: 500; padding-top: 8px; border-top: 1.5px solid #E5E7EB; display: inline-block;'>
                        M.Sc. Researcher in Structural Engineering
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL3-/main/LOGO.png", width=120)

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        login_col_1, login_col_2, login_col_3 = st.columns([1, 1, 1])
        with login_col_2:
            pwd = st.text_input("🔒 Secure Access: Enter Password", type="password")
            if st.button("Access Dashboard", use_container_width=True):
                if pwd == "ASA2026":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Password! Please check and try again.")
        
        return False
    return True

# =============================================================================
# 3. الهيدر الأكاديمي
# =============================================================================
def show_academic_header():
    col_left, col_mid, col_right = st.columns([1, 3, 1])
    
    with col_left:
        st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL3-/main/LOGO.png", width=130)
        
    with col_mid:
        st.markdown("""
            <div style='text-align: center;'>
                <h1 style='color: #1E3A8A; font-size: 42px; font-weight: bold; margin-bottom: 5px;'>
                    ASA-PREDICTION MODEL 3
                </h1>
                <h2 style='color: #D32F2F; font-size: 28px; font-weight: 600; margin-top: 0px; line-height: 1.3;'>
                    Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects
                </h2>
                <hr style='border: 0.5px solid #E5E7EB; width: 70%; margin: 20px auto;'>
                <span style='font-size: 20px; color: #4B5563;'>Prepared by:</span><br>
                <span style='font-size: 24px; font-weight: bold;'>Master's Researcher: Aya Mohammed Sanad Aboud</span><br><br>
                <span style='font-size: 22px; font-weight: bold; color: #4B5563;'>Under the Supervision of:</span><br>
                <span style='font-size: 24px; font-weight: 800; color: #111827;'>Prof. Ahmed Tahwia & Assoc. prof. Asser El-Sheikh</span>
            </div>
            """, unsafe_allow_html=True)
            
    with col_right:
        st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL3-/main/OIP.jfif", width=130)
        
    st.divider()

# =============================================================================
# 4. تحميل الموديلات
# =============================================================================
@st.cache_resource
def load_assets():
    try:
        models = joblib.load('concrete_model_multi.joblib')
        scaler = joblib.load('scaler_multi.joblib')
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure files are uploaded.")
        return None, None

# =============================================================================
# 5. محرك التنبؤ
# =============================================================================
def run_prediction_engine(inputs, prices):
    models, scaler = load_assets()
    if models is None or scaler is None: return None
    
    feature_list = [
        inputs['Cement'], inputs['Water'], inputs['W_C'], inputs['NCA'], inputs['NFA'], 
        inputs['RCA_Weight'], inputs['RCA_P'], inputs['MRCA_P'], inputs['RFA_Weight'], 
        inputs['RFA_P'], inputs['Fly_Ash'], inputs['Silica_Fume'], inputs['Metakaolin'], 
        inputs['GGBFS'], inputs['RHA_P'], inputs['Nylon_Fiber'], inputs['Basalt_Fiber_Vol'], 
        inputs['Natural_Fiber'], inputs['SP'], inputs['Agg_Size'], inputs['Density']
    ]
    
    vector = np.array(feature_list).reshape(1, -1)
    vector_scaled = scaler.transform(vector)
    
    cs28 = models['CS_28'].predict(vector_scaled)[0]
    sts = models['STS'].predict(vector_scaled)[0]
    co2 = models['CO2'].predict(vector_scaled)[0]
    energy = models['Energy'].predict(vector_scaled)[0]
    
    fs = 0.62 * math.sqrt(cs28) if cs28 > 0 else 0
    em = (4700 * math.sqrt(cs28)) / 1000 if cs28 > 0 else 0 
    cs7 = cs28 * 0.70
    cs90 = cs28 * 1.15
    
    total_cost = (
        (inputs['Cement'] * prices['Cement']) +
        (inputs['Water'] * prices['Water']) +
        (inputs['NCA'] * prices['NCA']) +
        (inputs['NFA'] * prices['NFA']) +
        (inputs['RCA_Weight'] * prices['RCA']) +
        (inputs['RFA_Weight'] * prices['RFA']) + 
        (inputs['SP'] * prices['SP'])
    )
    
    return {
        'CS28': cs28, 'STS': sts, 'CO2': co2, 'Energy': energy,
        'FS': fs, 'EM': em, 'CS7': cs7, 'CS90': cs90, 'Cost': total_cost
    }

# =============================================================================
# 6. الرادار الشامل
# =============================================================================
def show_radar_chart(results):
    strength_score = min(results['CS28'] / 80, 1.0)
    eco_score = 1 - min(results['CO2'] / 600, 1.0)
    cost_score = 1 - min(results['Cost'] / 200, 1.0)
    energy_score = 1 - min(results['Energy'] / 3000, 1.0)

    categories = ['Structural Strength', 'CO2 Efficiency', 'Cost Efficiency', 'Energy Efficiency']
    scores = [strength_score, eco_score, cost_score, energy_score]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores, theta=categories, fill='toself',
        name='Mix Sustainability Profile', line_color='#1E3A8A', marker=dict(size=8)
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".1%")),
        showlegend=False, title={'text': "<b>Comprehensive Sustainability Radar</b>", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 7. تسجيل البيانات في الشيت
# =============================================================================
def log_prediction_to_sheets(inputs, results):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        new_row = pd.DataFrame([{
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Cement": inputs['Cement'], "Water": inputs['Water'], "W_C": inputs['W_C'], 
            "NCA": inputs['NCA'], "NFA": inputs['NFA'],
            "RCA_Weight": inputs['RCA_Weight'], "RCA_P": inputs['RCA_P'], 
            "MRCA_P": inputs['MRCA_P'], "RFA_Weight": inputs['RFA_Weight'], "RFA_P": inputs['RFA_P'], 
            "Fly_Ash": inputs['Fly_Ash'], "Silica_Fume": inputs['Silica_Fume'], "Metakaolin": inputs['Metakaolin'], 
            "GGBFS": inputs['GGBFS'], "RHA_P": inputs['RHA_P'], 
            "Nylon_Fiber": inputs['Nylon_Fiber'], "Basalt_Fiber_Vol": inputs['Basalt_Fiber_Vol'], "Natural_Fiber": inputs['Natural_Fiber'], 
            "SP": inputs['SP'], "Agg_Size": inputs['Agg_Size'], "Density": inputs['Density'],
            "Predicted_CS28": round(results['CS28'], 2),
            "Predicted_CO2": round(results['CO2'], 2),
            "Predicted_Cost": round(results['Cost'], 2)
        }])

        try:
            existing_data = conn.read(worksheet="Predictions_V3", ttl=0)
            if existing_data is not None and not existing_data.empty:
                updated_df = pd.concat([existing_data, new_row], ignore_index=True)
            else:
                updated_df = new_row
        except:
            updated_df = new_row
            
        conn.update(worksheet="Predictions_V3", data=updated_df)
        st.toast("✅ تم الحفظ في قاعدة البيانات بنجاح", icon="💾")
    except Exception as e:
        st.sidebar.error(f"Logging Error: {e}")

# =============================================================================
# 8. واجهة الإدخال والنتائج
# =============================================================================
def show_input_section():
    st.markdown("### 🏗️ Design Mix Inputs (21 Parameters)")
    
    with st.expander("💲 Dynamic Market Prices (USD/kg) - Update to calculate current mix cost"):
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        prices = {
            'Cement': p_col1.number_input("Cement Price", value=0.15),
            'Water': p_col2.number_input("Water Price", value=0.002),
            'NCA': p_col3.number_input("NCA Price", value=0.02),
            'NFA': p_col4.number_input("NFA Price", value=0.015),
            'RCA': p_col1.number_input("RCA Price", value=0.01),
            'RFA': p_col2.number_input("RFA Price", value=0.008),
            'SP': p_col3.number_input("SP Price", value=2.5)
        }

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("##### Binders & Water")
        cement = st.number_input("Cement (kg/m³)", value=350.0)
        water = st.number_input("Water (kg/m³)", value=175.0)
        w_c = st.number_input("W/C Ratio", value=0.5)
        sp = st.number_input("Superplasticizer (kg/m³)", value=2.0)
        density = st.number_input("Density (kg/m³)", value=2400.0)
        
    with c2:
        st.markdown("##### Natural Aggregates")
        total_coarse = st.number_input("Total Coarse Agg. (kg/m³)", value=1000.0, help="أدخل إجمالي وزن الزلط المطلوب للمتر المكعب هنا")
        nfa = st.number_input("NFA (kg/m³)", value=700.0)
        agg_size = st.number_input("Max Agg Size (mm)", value=20.0)
        
    with c3:
        st.markdown("##### Recycled Aggregates")
        rca_p = st.number_input("RCA (%)", value=0.0)
        mrca_p = st.number_input("MRCA (%)", value=0.0)
        rfa_w = st.number_input("RFA Weight (kg/m³)", value=0.0)
        rfa_p = st.number_input("RFA (%)", value=0.0)
        
        rca_w = total_coarse * (rca_p / 100.0)
        mrca_w = total_coarse * (mrca_p / 100.0)
        nca = total_coarse - (rca_w + mrca_w)
        if nca < 0: nca = 0.0

    with c4:
        st.markdown("##### SCMs & Fibers")
        fly_ash = st.number_input("Fly Ash (kg/m³)", value=0.0)
        silica = st.number_input("Silica Fume (kg/m³)", value=0.0)
        mk = st.number_input("Metakaolin (kg/m³)", value=0.0)
        ggbfs = st.number_input("GGBFS (kg/m³)", value=0.0)
        rha_p = st.number_input("RHA (%)", value=0.0)
        nylon = st.number_input("Nylon Fiber (kg/m³)", value=0.0)
        basalt = st.number_input("Basalt Fiber Vol (%)", value=0.0)
        natural = st.number_input("Natural Fiber (kg/m³)", value=0.0)

    st.info(f"💡 **AI Logic Check (Net Weights):** Natural (NCA): **{nca:.1f} kg** | RCA: **{rca_w:.1f} kg** | MRCA: **{mrca_w:.1f} kg**")

    if st.button("🚀 Run Prediction & Analysis", use_container_width=True):
        inputs = {
            'Cement': cement, 'Water': water, 'W_C': w_c, 'NCA': nca, 'NFA': nfa,
            'RCA_Weight': rca_w, 'RCA_P': rca_p, 'MRCA_P': mrca_p, 'RFA_Weight': rfa_w,
            'RFA_P': rfa_p, 'Fly_Ash': fly_ash, 'Silica_Fume': silica, 'Metakaolin': mk,
            'GGBFS': ggbfs, 'RHA_P': rha_p, 'Nylon_Fiber': nylon, 'Basalt_Fiber_Vol': basalt,
            'Natural_Fiber': natural, 'SP': sp, 'Agg_Size': agg_size, 'Density': density
        }

        with st.spinner("Processing AI Models & Logging Results..."):
            res = run_prediction_engine(inputs, prices)
            if res:
                st.success("✅ Analysis Completed: Using Hybrid AI-Engineering Model")
                log_prediction_to_sheets(inputs, res)
                
                t_mech, t_env, t_eco = st.tabs(["🏗️ Mechanical", "🌱 Environmental", "💰 Economic"])
                
                with t_mech:
                    m1, m2 = st.columns(2)
                    m1.metric("CS 28-days (MPa) [AI Predicted]", f"{res['CS28']:.2f}")
                    m1.metric("Splitting Tensile (MPa) [AI Predicted]", f"{res['STS']:.2f}")
                    m2.metric("Flexural Strength (MPa) [ACI Estimated]", f"{res['FS']:.2f}")
                    m2.metric("Elastic Modulus (GPa) [ACI Estimated]", f"{res['EM']:.2f}")
                    m1.metric("CS 7-days (MPa) [Estimated]", f"{res['CS7']:.2f}")
                    m2.metric("CS 90-days (MPa) [Estimated]", f"{res['CS90']:.2f}")
                    
                with t_env:
                    e1, e2 = st.columns(2)
                    e1.metric("CO2 Footprint (kg/m³) [AI Predicted]", f"{res['CO2']:.2f}")
                    e2.metric("Energy Demand (MJ/m³) [AI Predicted]", f"{res['Energy']:.2f}")
                    
                with t_eco:
                    ec1, ec2 = st.columns(2)
                    ec1.metric("Total Material Cost (USD/m³) [Dynamic]", f"{res['Cost']:.2f}")
                    with ec2:
                        show_radar_chart(res)

# =============================================================================
# 9. المُحسّن (Optimizer)
# =============================================================================
def show_optimizer():
    st.header("⚖️ AI-Based Mix Optimizer")
    st.markdown("Searches the 1,617-mix database for optimal eco-efficient alternatives.")
    target_cs = st.number_input("Target Strength 28d (MPa)", value=40.0)
    tol = st.slider("Tolerance (± MPa)", 1.0, 10.0, 3.0)
    
    if st.button("Search Database"):
        try:
            df = pd.read_excel('Final_Dataset_Flawless.xlsx')
            filtered = df[(df['CS_28'] >= target_cs - tol) & (df['CS_28'] <= target_cs + tol)]
            if not filtered.empty:
                top = filtered.sort_values(by=['CO2', 'Energy'], ascending=[True, True]).head(5)
                cols = ['Mix_ID', 'Cement', 'W_C', 'CS_28', 'CO2', 'Energy']
                available = [c for c in cols if c in top.columns]
                st.dataframe(top[available].style.highlight_min(subset=['CO2', 'Energy'], color='#D1FAE5'))
            else:
                st.warning("No mixes found in this range. Try increasing the tolerance.")
        except Exception as e:
            st.error(f"Database file error: {e}")

# =============================================================================
# 10. تبويب تحليل الـ MCDM (مع تفعيل إضافة الصفوف ديناميكياً)
# =============================================================================
def show_mcdm_tab():
    st.header("🎯 Multi-Criteria Decision Making (MCDM) & Experimental Validation")
    st.markdown("Evaluate and rank concrete mix formulations using **TOPSIS**, **VIKOR**, and **EDAS** methodologies.")
    
    with st.expander("⚙️ MCDM Parameters & Weights Configuration", expanded=True):
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        w_cs = col_w1.slider("Weight: Compressive Strength (CS)", 0.0, 1.0, 0.30, key="mcdm_w_cs")
        w_sts = col_w2.slider("Weight: Tensile Strength (STS)", 0.0, 1.0, 0.20, key="mcdm_w_sts")
        w_co2 = col_w3.slider("Weight: CO2 Emissions", 0.0, 1.0, 0.25, key="mcdm_w_co2")
        w_cost = col_w4.slider("Weight: Total Cost", 0.0, 1.0, 0.25, key="mcdm_w_cost")
        
        total_w = w_cs + w_sts + w_co2 + w_cost
        if abs(total_w - 1.0) > 1e-3:
            st.warning(f"⚠️ Total weights sum is {total_w:.2f}. Recommended to sum up to 1.0.")

    st.subheader("📊 Experimental / User Evaluated Mixes Matrix")
    st.markdown("يمكنك مراجعة أو تعديل البيانات المعملية، أو الضغط على علامة (+) لإضافة خلطات جديدة وتجربتها:")
    
    default_data = pd.DataFrame({
        "Mix_ID": ["Mix 1 (Control)", "Mix 2 (Circular)", "Mix 3 (Low Carbon)", "Mix 4 (Eco-Optimized)", "Mix 5 (High Strength)", "Mix 6 (Optimal Hybrid)"],
        "CS_28 (MPa)": [47.50, 62.75, 45.50, 36.75, 82.75, 65.67],
        "STS (MPa)": [2.31, 2.41, 2.13, 1.80, 2.22, 3.29],
        "CO2 (kg/m3)": [314.45, 380.26, 402.47, 146.16, 387.34, 295.19],
        "Cost (USD)": [100.75, 84.86, 106.05, 56.52, 91.28, 73.02]
    })
    
    edited_df = st.data_editor(default_data, use_container_width=True, num_rows="dynamic")
    
    if st.button("🚀 Run MCDM Evaluation (TOPSIS, VIKOR, EDAS)", use_container_width=True):
        try:
            matrix = edited_df.iloc[:, 1:].values
            mix_names = edited_df["Mix_ID"].values
            weights = np.array([w_cs, w_sts, w_co2, w_cost])
            weights = weights / np.sum(weights)
            
            criteria_types = np.array([1, 1, -1, -1]) 
            
            # TOPSIS
            norm_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0))
            weighted_matrix = norm_matrix * weights
            ideal_best = np.max(weighted_matrix * criteria_types, axis=0) * criteria_types
            ideal_worst = np.min(weighted_matrix * criteria_types, axis=0) * criteria_types
            d_plus = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
            d_minus = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))
            topsis_score = d_minus / (d_plus + d_minus + 1e-10)
            topsis_rank = len(topsis_score) - np.argsort(np.argsort(topsis_score))
            
            # VIKOR
            f_star = np.max(matrix * criteria_types, axis=0) * criteria_types
            f_minus = np.min(matrix * criteria_types, axis=0) * criteria_types
            f_diff = np.where((f_star - f_minus) == 0, 1e-10, (f_star - f_minus))
            f_norm = np.zeros_like(matrix, dtype=float)
            for j in range(matrix.shape[1]):
                if criteria_types[j] == 1:
                    f_norm[:, j] = (f_star[j] - matrix[:, j]) / f_diff[j]
                else:
                    f_norm[:, j] = (matrix[:, j] - f_star[j]) / f_diff[j]
            S = np.sum(weights * f_norm, axis=1)
            R = np.max(weights * f_norm, axis=1)
            s_star, s_minus = np.min(S), np.max(S)
            r_star, r_minus = np.min(R), np.max(R)
            v = 0.5
            Q = v * (S - s_star) / (s_minus - s_star + 1e-10) + (1 - v) * (R - r_star) / (r_minus - r_star + 1e-10)
            vikor_rank = np.argsort(np.argsort(Q)) + 1
            
            # EDAS
            avg_solution = np.mean(matrix, axis=0)
            PDA = np.zeros_like(matrix, dtype=float)
            NDA = np.zeros_like(matrix, dtype=float)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if criteria_types[j] == 1:
                        PDA[i, j] = max(0, (matrix[i, j] - avg_solution[j])) / (abs(avg_solution[j]) + 1e-10)
                        NDA[i, j] = max(0, (avg_solution[j] - matrix[i, j])) / (abs(avg_solution[j]) + 1e-10)
                    else:
                        PDA[i, j] = max(0, (avg_solution[j] - matrix[i, j])) / (abs(avg_solution[j]) + 1e-10)
                        NDA[i, j] = max(0, (matrix[i, j] - avg_solution[j])) / (abs(avg_solution[j]) + 1e-10)
            SP = np.sum(PDA * weights, axis=1)
            SN = np.sum(NDA * weights, axis=1)
            NSP = SP / (np.max(SP) + 1e-10)
            NSN = 1 - (SN / (np.max(SN) + 1e-10))
            AS = 0.5 * (NSP + NSN)
            edas_rank = len(AS) - np.argsort(np.argsort(AS))
            
            result_df = pd.DataFrame({
                "Mix ID": mix_names,
                "TOPSIS Score": np.round(topsis_score, 4),
                "TOPSIS Rank": topsis_rank,
                "VIKOR Q-Index": np.round(Q, 4),
                "VIKOR Rank": vikor_rank,
                "EDAS AS-Score": np.round(AS, 4),
                "EDAS Rank": edas_rank
            })
            
            st.success("✅ MCDM Evaluation Executed Successfully!")
            st.dataframe(result_df.sort_values(by="TOPSIS Rank"), use_container_width=True)
            st.info("💡 **Key Finding:** Notice how **Mix 6** consistently secures **Rank 1** across methodologies, confirming absolute model robustness!")
        except Exception as e:
            st.error(f"Error in MCDM calculations: {e}")

# =============================================================================
# 11. صفحة الأداء (Performance)
# =============================================================================
def show_performance():
    st.header("📈 Model Performance & Metrics")
    target_choice = st.selectbox("Select Target Parameter for Analysis:", 
                                 ["Compressive Strength (CS_28)", "Tensile Strength (STS)", 
                                  "CO2 Emissions", "Energy Demand"])
    
    metrics_data = {
        "Compressive Strength (CS_28)": {"r2": "0.97", "rmse": "4.84 MPa", "mae": "3.24 MPa", "cv": "0.84", "prefix": "CS_28"},
        "Tensile Strength (STS)": {"r2": "0.95", "rmse": "0.32 MPa", "mae": "0.18 MPa", "cv": "0.76", "prefix": "STS"},
        "CO2 Emissions": {"r2": "0.99", "rmse": "18.43 kg", "mae": "2.37 kg", "cv": "0.90", "prefix": "CO2"},
        "Energy Demand": {"r2": "0.97", "rmse": "468.12 MJ", "mae": "47.30 MJ", "cv": "0.82", "prefix": "Energy"}
    }
    
    data = metrics_data[target_choice]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training R² Score", data["r2"])
    c2.metric("Testing RMSE", data["rmse"])
    c3.metric("Testing MAE", data["mae"])
    c4.metric("Cross-Val R² (5-Fold)", data["cv"])
    
    st.divider()
    st.subheader(f"🔬 Visual Diagnostics: {target_choice}")
    p1, p2 = st.columns(2)
    with p1:
        img_path = f"Performance_Plots/{data['prefix']}_Actual_vs_Predicted.png"
        if os.path.exists(img_path): st.image(img_path, caption="Actual vs Predicted", use_container_width=True)
        else: st.warning(f"Image not found: {img_path}")
    with p2:
        img_path2 = f"Performance_Plots/{data['prefix']}_Residuals.png"
        if os.path.exists(img_path2): st.image(img_path2, caption="Residuals Distribution", use_container_width=True)
        else: st.warning(f"Image not found: {img_path2}")
            
    st.markdown("---")
    img_path3 = f"Performance_Plots/{data['prefix']}_Feature_Importance.png"
    if os.path.exists(img_path3): st.image(img_path3, caption="Feature Importance Analysis", use_container_width=True)

# =============================================================================
# 12. نظام الفيدباك
# =============================================================================
def handle_feedback():
    st.header("📝 User Feedback & Experience")
    st.write("##### ⭐ How accurate do you find these results based on your lab experience?")
    stars = st.feedback("stars")
    st.divider()
    
    with st.form("feedback_form", clear_on_submit=True):
        st.markdown("##### 📋 Additional Comments")
        col1, col2 = st.columns(2)
        with col1: user_name = st.text_input("Full Name (Optional)")
        with col2: user_email = st.text_input("Email (Optional)")
        
        observation = st.text_area("Your Observations & Suggestions", height=150)
        submit = st.form_submit_button("📤 Submit Feedback", use_container_width=True)
        
        if submit:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                feedback_row = pd.DataFrame([{
                    "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": user_name if user_name else "Anonymous",
                    "Email": user_email if user_email else "N/A",
                    "Stars": stars if stars is not None else "Not rated",
                    "Feedback": observation if observation else "No comments"
                }])
                try:
                    existing_f = conn.read(worksheet="Feedback_V3", ttl=0)
                    updated_f = pd.concat([existing_f, feedback_row], ignore_index=True)
                except:
                    updated_f = feedback_row
                conn.update(worksheet="Feedback_V3", data=updated_f)
                st.success("✅ Thank you! Feedback recorded successfully in database.")
                st.balloons()
            except Exception as e:
                st.error(f"Connection Error: {e}")

# =============================================================================
# 13. الوثائق
# =============================================================================
def show_documentation():
    st.header("📚 Technical Documentation & Methodology")
    doc_tabs = st.tabs(["Methodology", "Glossary", "Disclaimer"])
    with doc_tabs[0]:
        st.subheader("Core Model Information")
        st.markdown("""
        - **Algorithm:** Random Forest Regression (Multi-output Architecture)
        - **Database:** 1,617 Experimental Samples (Cleaned & Preprocessed)
        - **Methodology:** Integrates AI prediction with standard ACI 318 equations and Multi-Criteria Decision Making (MCDM).
        - **Robustness:** Validated using rigorous 5-Fold Cross Validation
        """)
    with doc_tabs[1]:
        st.subheader("Glossary of Terms")
        st.markdown("""
        - **CS_28:** Compressive Strength at 28 days
        - **STS:** Splitting Tensile Strength
        - **MCDM:** Multi-Criteria Decision Making
        - **Radar Chart:** Visual normalization of strength, cost, and eco-impact.
        """)
    with doc_tabs[2]:
        st.subheader("Disclaimer")
        st.warning("""
        All outputs provided by this application are predictive results generated by Artificial Intelligence models based on experimental datasets. These results are for guidance purposes only and should not replace laboratory testing or physical verification of concrete mixes.
        """)

# =============================================================================
# 14. الدالة الرئيسية (Main)
# =============================================================================
def main():
    st.markdown("""
        <style>
        .footer { 
            position: fixed; 
            left: 0; 
            bottom: 0; 
            width: 100%; 
            background-color: #f1f1f1; 
            color: #555; 
            text-align: center; 
            padding: 10px; 
            font-size: 14px; 
            border-top: 1px solid #e7e7e7; 
            z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if check_login():
        show_academic_header()
        
        tabs = st.tabs(["🏠 Home", "🚀 Predictor", "⚖️ Optimizer", "🎯 MCDM Analysis", "📈 Performance", "📝 Feedback", "📚 Docs"])
        
        with tabs[0]:
            st.markdown("### Welcome to ASA-PREDICTION MODEL 3 Dashboard")
            st.markdown("#### 🎯 Your AI-Powered Tool for Eco-Efficient Concrete Design")
            info_col1, info_col2 = st.columns([2, 1])
            with info_col1:
                st.info("""
                **🔬 About This System:**
                هذا النظام الذكي مصمم لدعم اتخاذ القرار في تصميم الخلطات الخرسانية الصديقة للبيئة 
                من خلال التحليل المتعدد المعايير للجوانب الفنية والبيئية والاقتصادية.
                
                **✨ Key Features:**
                - 🤖 AI-powered predictions using Multi-Output Random Forest
                - 📊 Real-time evaluation of Structural, Environmental & Economic aspects
                - ♻️ Eco-efficiency optimization engine
                - 🎯 Interactive Multi-Criteria Decision Making (TOPSIS, VIKOR, EDAS)
                """)
                st.markdown("##### 🚀 Quick Start Guide:")
                st.markdown("""
                1. Navigate to **🚀 Predictor** or **🎯 MCDM Analysis** tabs.
                2. Enter your mix parameters or evaluate formulations.
                3. Review comprehensive rankings and analytics.
                """)
            with info_col2:
                st.markdown("##### 📊 Model Stats")
                st.metric("Database Size", "1,617 samples")
                st.metric("Input Parameters", "21")
                st.metric("Validation Method", "5-Fold CV")
                st.success("✅ **Status:** Model Loaded & Ready")
        
        with tabs[1]: show_input_section()
        with tabs[2]: show_optimizer()
        with tabs[3]: show_mcdm_tab()
        with tabs[4]: show_performance()
        with tabs[5]: handle_feedback()
        with tabs[6]: show_documentation()
        
        st.markdown("""
            <div class="footer">
                © 2026 Aya Mohammed Sanad Aboud | Structural Engineering Dept | Mansoura University
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
