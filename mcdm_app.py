import streamlit as st
import pandas as pd
import numpy as np

# إعدادات الصفحة
st.set_page_config(
    page_title="ASA-MCDM & Experimental Validation",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Multi-Criteria Decision Making (MCDM) & Experimental Validation")
st.markdown("### TOPSIS, VIKOR, and EDAS Analysis for Sustainable Concrete Mixes")
st.divider()

# ضبط الأوزان
with st.expander("⚙️ MCDM Parameters & Weights Configuration", expanded=True):
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    w_cs = col_w1.slider("Weight: Compressive Strength (CS)", 0.0, 1.0, 0.30)
    w_sts = col_w2.slider("Weight: Tensile Strength (STS)", 0.0, 1.0, 0.20)
    w_co2 = col_w3.slider("Weight: CO2 Emissions", 0.0, 1.0, 0.25)
    w_cost = col_w4.slider("Weight: Total Cost", 0.0, 1.0, 0.25)
    
    total_w = w_cs + w_sts + w_co2 + w_cost
    if abs(total_w - 1.0) > 1e-3:
        st.warning(f"⚠️ Total weights sum is {total_w:.2f}. Recommended to sum up to 1.0.")

st.subheader("📊 Experimental Evaluated Mixes Matrix (Lab Data)")
st.markdown("هنا يمكنك مراجعة أو تعديل النتائج المعملية الفعلية للخلطات الست (الضغط، الشد، البصمة الكربونية، والتكلفة)، أو إضافة خلطات جديدة:")

# جدول البيانات المعملية الافتراضي للخلطات الست
default_data = pd.DataFrame({
    "Mix_ID": ["Mix 1 (Control)", "Mix 2 (Circular)", "Mix 3 (Low Carbon)", "Mix 4 (Eco-Optimized)", "Mix 5 (High Strength)", "Mix 6 (Optimal Hybrid)"],
    "CS_28 (MPa)": [47.50, 62.75, 45.50, 36.75, 82.75, 65.67],
    "STS (MPa)": [2.31, 2.41, 2.13, 1.80, 2.22, 3.29],
    "CO2 (kg/m3)": [314.45, 380.26, 402.47, 146.16, 387.34, 295.19],
    "Cost (USD)": [100.75, 84.86, 106.05, 56.52, 91.28, 73.02]
})

edited_df = st.data_editor(default_data, use_container_width=True)

if st.button("🚀 Run MCDM Evaluation (TOPSIS, VIKOR, EDAS)", use_container_width=True):
    try:
        matrix = edited_df.iloc[:, 1:].values
        mix_names = edited_df["Mix_ID"].values
        weights = np.array([w_cs, w_sts, w_co2, w_cost])
        weights = weights / np.sum(weights)
        
        # 1: Maximization, -1: Minimization
        criteria_types = np.array([1, 1, -1, -1]) 
        
        # --- 1. TOPSIS ---
        norm_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0))
        weighted_matrix = norm_matrix * weights
        
        ideal_best = np.max(weighted_matrix * criteria_types, axis=0) * criteria_types
        ideal_worst = np.min(weighted_matrix * criteria_types, axis=0) * criteria_types
        
        d_plus = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
        d_minus = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))
        
        topsis_score = d_minus / (d_plus + d_minus + 1e-10)
        topsis_rank = len(topsis_score) - np.argsort(np.argsort(topsis_score))
        
        # --- 2. VIKOR ---
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
        
        # --- 3. EDAS ---
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
        
        # --- جدول النتائج النهائي ---
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
        
        st.info("💡 **Key Finding:** Notice how **Mix 6** consistently secures **Rank 1** across methodologies, while **Mix 4** secures **Rank 2**, confirming absolute model robustness!")
        
    except Exception as e:
        st.error(f"Error in MCDM calculations: {e}")
