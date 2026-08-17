import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="จำแนกเห็ดกินได้ / มีพิษ", page_icon="🍄", layout="wide")

# ---------- load artifacts ----------
FEATURE_COLUMNS = json.load(open("feature_columns.json", encoding="utf-8"))
CATEGORY_OPTIONS = json.load(open("category_options.json", encoding="utf-8"))
MODEL_COMPARISON = pd.read_json("model_comparison.json")
model = joblib.load("dt_model.pkl")

FEATURE_LABELS_TH = {
    "cap-shape": "รูปทรงหมวก (cap-shape)",
    "cap-surface": "พื้นผิวหมวก (cap-surface)",
    "cap-color": "สีหมวก (cap-color)",
    "bruises": "รอยช้ำเมื่อสัมผัส (bruises)",
    "odor": "กลิ่น (odor)",
    "gill-attachment": "ลักษณะการติดของครีบ (gill-attachment)",
    "gill-spacing": "ระยะห่างของครีบ (gill-spacing)",
    "gill-size": "ขนาดครีบ (gill-size)",
    "gill-color": "สีครีบ (gill-color)",
    "stalk-shape": "รูปทรงก้าน (stalk-shape)",
    "stalk-root": "ลักษณะรากก้าน (stalk-root)",
    "stalk-surface-above-ring": "พื้นผิวก้านเหนือวงแหวน",
    "stalk-surface-below-ring": "พื้นผิวก้านใต้วงแหวน",
    "stalk-color-above-ring": "สีก้านเหนือวงแหวน",
    "stalk-color-below-ring": "สีก้านใต้วงแหวน",
    "veil-color": "สีเยื่อหุ้ม (veil-color)",
    "ring-number": "จำนวนวงแหวน (ring-number)",
    "ring-type": "ชนิดวงแหวน (ring-type)",
    "spore-print-color": "สีของสปอร์พิมพ์ (spore-print-color)",
    "population": "ลักษณะการอยู่รวมกัน (population)",
    "habitat": "แหล่งที่อยู่ (habitat)",
}

# ---------- header ----------
col_photo, col_info = st.columns([1, 5])
with col_photo:
    st.image("profile.jpg", width=140)
with col_info:
    st.markdown("### ณรงค์ศักดิ์ ประเสริฐศิริสร")
    st.caption("รหัสนักศึกษา 664245033  ·  หมู่เรียน 66/44  ·  วิชา ระบบฐานข้อมูลขั้นสูง")

st.title("🍄 ระบบจำแนกเห็ดกินได้ / มีพิษ ด้วย Decision Tree")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. โจทย์ & Dataset",
    "2. Data Preprocessing",
    "3. ทฤษฎีโมเดล",
    "4. เปรียบเทียบโมเดล",
    "5. ทำนายผล",
])

# ============================================================
# TAB 1 — โจทย์ปัญหาและ Dataset
# ============================================================
with tab1:
    st.header("โจทย์ปัญหา")
    st.write(
        "ทำนายว่าเห็ดชนิดหนึ่ง **กินได้ (edible)** หรือ **มีพิษ (poisonous)** "
        "จากลักษณะทางกายภาพที่สังเกตได้ด้วยตาเปล่า เช่น รูปทรงและสีของหมวกเห็ด "
        "กลิ่น ลักษณะครีบ ก้าน วงแหวน และสปอร์ โดยไม่ต้องอาศัยผู้เชี่ยวชาญด้านพฤกษศาสตร์"
    )

    st.header("เหตุผลที่เลือกชุดข้อมูลนี้")
    st.markdown(
        """
1. เป็นปัญหา **Binary Classification** ที่มีความหมายในชีวิตจริงอย่างชัดเจน — ช่วยลดความเสี่ยงจากการเก็บเห็ดพิษมารับประทาน
2. ตัวแปรทั้งหมดเป็นข้อมูลเชิงหมวดหมู่ **(categorical) ล้วน** ต่างจาก 6 ชุดข้อมูลก่อนหน้าที่ส่วนใหญ่เป็นตัวเลข — เหมาะสำหรับสาธิตการทำ One-Hot Encoding และการทำงานของ Decision Tree กับข้อมูลเชิงหมวดหมู่โดยเฉพาะ
3. เป็นชุดข้อมูลคลาสสิกจาก **UCI Machine Learning Repository (Mushroom Data Set)** มีจำนวนตัวอย่างมากพอสมควรและถูกใช้อ้างอิงในวงการ ML มายาวนาน
        """
    )

    st.header("ภาพรวม Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนตัวอย่าง", "8,124")
    c2.metric("จำนวนตัวแปรต้น (ก่อนตัด)", "22")
    c3.metric("คลาสเป้าหมาย", "2 (edible / poisonous)")

    st.caption("ตัวอย่างข้อมูลดิบ (ถอดรหัสจากตัวย่อ UCI แล้ว):")
    preview = pd.DataFrame({
        "cap-shape": ["convex", "convex", "bell"],
        "cap-color": ["brown", "yellow", "white"],
        "odor": ["pungent", "almond", "anise"],
        "gill-color": ["black", "black", "brown"],
        "habitat": ["urban", "grasses", "meadows"],
        "class": ["poisonous", "edible", "edible"],
    })
    st.dataframe(preview, use_container_width=True)
    st.caption("แหล่งที่มา: UCI Machine Learning Repository — Mushroom Data Set")

# ============================================================
# TAB 2 — Data Preprocessing
# ============================================================
with tab2:
    st.header("ขั้นตอนการเตรียมข้อมูล")

    st.subheader("1) ตรวจสอบค่าที่ขาดหาย (Missing Values)")
    st.write(
        "ตัวแปร **stalk-root** มีค่าที่ไม่ระบุ (`?`) อยู่ **2,480 จาก 8,124 แถว (~30.5%)** "
        "เนื่องจากค่าที่ขาดหายอาจไม่ได้เกิดขึ้นแบบสุ่มล้วน ๆ (อาจสื่อถึงลักษณะบางอย่างของเห็ดกลุ่มนั้น) "
        "จึงเลือก **เก็บไว้เป็นอีกหนึ่งหมวดหมู่ (\"missing\")** แทนการลบทั้งแถวทิ้งหรือเติมด้วยค่าฐานนิยม "
        "เพื่อไม่ให้สูญเสียข้อมูลจำนวนมาก"
    )

    st.subheader("2) ตัดตัวแปรที่ไม่มีนัยสำคัญ")
    st.write(
        "ตัวแปร **veil-type** มีค่าเดียวตลอดทั้งชุดข้อมูล (ทุกแถวเป็น `partial`) "
        "จึงไม่มีผลต่อการจำแนกและถูกตัดออก เหลือตัวแปรต้น **21 ตัว**"
    )

    st.subheader("3) แปลงข้อมูลหมวดหมู่เป็นตัวเลข (One-Hot Encoding)")
    st.write(
        "เนื่องจากไม่มีตัวแปรใดมีลำดับ (ordinal) ที่แท้จริง — เช่น สีหรือกลิ่นไม่มี \"มาก-น้อย\" "
        "จึงใช้ **One-Hot Encoding** (`pd.get_dummies`) กับตัวแปรทั้ง 21 ตัว "
        "ทำให้ได้ตัวแปรนำเข้าโมเดลทั้งหมด **116 คอลัมน์**"
    )

    st.subheader("4) แบ่งชุดข้อมูล Train / Test")
    st.write(
        "แบ่งข้อมูลแบบ **Stratified 80/20** เพื่อรักษาสัดส่วนของคลาส edible/poisonous "
        "ให้ใกล้เคียงกันทั้งชุดฝึกและชุดทดสอบ"
    )

    st.info("สรุป: 8,124 แถว × 22 ตัวแปร  →  ตัด veil-type  →  21 ตัวแปร  →  One-Hot Encoding  →  116 คอลัมน์  →  แบ่ง Train 6,499 / Test 1,625")

# ============================================================
# TAB 3 — ทฤษฎีโมเดล (Decision Tree)
# ============================================================
with tab3:
    st.header("ทฤษฎีของโมเดล: Decision Tree")
    st.write(
        "**Decision Tree** จำแนกข้อมูลด้วยการแบ่งข้อมูลซ้ำ ๆ ตามเงื่อนไขของตัวแปรทีละขั้น "
        "(เหมือนต้นไม้ที่แตกกิ่งก้าน) จนกระทั่งถึง **ใบ (leaf)** ที่ให้คำตอบสุดท้ายว่าเป็น edible หรือ poisonous"
    )

    st.subheader("เกณฑ์การแบ่งกิ่ง: Entropy / Information Gain")
    st.latex(r"H(S) = -\sum_{i} p_i \log_2 p_i")
    st.write(
        "ในแต่ละขั้น โมเดลจะพิจารณาทุกตัวแปรและเลือก **จุดแบ่งที่ทำให้ค่า Entropy (ความไม่บริสุทธิ์ของข้อมูล) "
        "หลังแบ่งลดลงมากที่สุด** หรือกล่าวอีกนัยหนึ่งคือให้ **Information Gain สูงสุด** "
        "แล้วทำซ้ำขั้นตอนนี้ไปเรื่อย ๆ ในแต่ละกิ่งย่อยจนกว่าจะถึงเงื่อนไขหยุด"
    )

    st.subheader("การจำกัดความลึกของต้นไม้ (max_depth = 6)")
    st.write(
        "หากปล่อยให้ต้นไม้แตกกิ่งได้ไม่จำกัด โมเดลจะ Overfit และแยกข้อมูลได้ 100% แต่ตีความยาก "
        "จึงจำกัด **max_depth = 6** เพื่อให้โมเดลยังคง **อธิบายเหตุผลการตัดสินใจได้ง่าย** "
        "ซึ่งสำคัญมากสำหรับงานที่เกี่ยวข้องกับความปลอดภัยในการบริโภค"
    )

    st.subheader("ตัวอย่างกฎการตัดสินใจจริงจากโมเดล (3 ระดับแรก)")
    try:
        with open("dt_rules_preview.txt", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    except FileNotFoundError:
        st.warning("ไม่พบไฟล์ dt_rules_preview.txt")

    st.subheader("ตัวแปรที่มีอิทธิพลต่อการทำนายมากที่สุด")
    st.image("feature_importance.png", use_container_width=True)
    st.caption(
        "ตัวแปร **odor (กลิ่น)** มีอิทธิพลต่อการทำนายมากที่สุดอย่างชัดเจน "
        "สอดคล้องกับความรู้ทั่วไปที่ว่ากลิ่นเป็นสัญญาณสำคัญของเห็ดพิษหลายชนิด"
    )

# ============================================================
# TAB 4 — เปรียบเทียบโมเดล
# ============================================================
with tab4:
    st.header("การประเมินและเปรียบเทียบโมเดล")
    st.write("เปรียบเทียบ Decision Tree (โมเดลหลัก) กับอีก 2 เทคนิคที่เคยใช้ในโปรเจกต์ก่อนหน้า บนชุดทดสอบเดียวกัน")

    display_df = MODEL_COMPARISON.copy()
    for col in ["accuracy", "precision", "recall", "f1"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
    display_df.columns = ["โมเดล", "Accuracy", "Precision", "Recall", "F1-score"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.image("comparison_chart.png", use_container_width=True)

    st.subheader("Confusion Matrix (Decision Tree)")
    cm = np.load("dt_confusion_matrix.npy")
    cm_df = pd.DataFrame(
        cm,
        index=["จริง: edible", "จริง: poisonous"],
        columns=["ทำนาย: edible", "ทำนาย: poisonous"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("สรุปผล")
    st.write(
        "ชุดข้อมูลนี้สามารถแยกประเภทได้เกือบสมบูรณ์แบบ เนื่องจากตัวแปร **odor** เพียงตัวเดียว "
        "ก็ให้ความแม่นยำสูงมากอยู่แล้ว ทำให้ทั้งสามโมเดลมีผลลัพธ์สูงใกล้เคียงกัน (>99%) "
        "โดย Random Forest และ KNN ทำได้แม่นยำ 100% บนชุดทดสอบ ขณะที่ Decision Tree "
        "(จำกัดความลึกเพื่อให้ตีความได้ง่าย) ทำได้ 99.88% แม้จะแม่นยำน้อยกว่าเล็กน้อย "
        "แต่ **Decision Tree ถูกเลือกเป็นโมเดลหลัก** เพราะสามารถอธิบายเหตุผลของการตัดสินใจแต่ละขั้นได้ชัดเจน "
        "ซึ่งสำคัญมากสำหรับงานที่เกี่ยวข้องกับความปลอดภัยในการบริโภค"
    )

# ============================================================
# TAB 5 — ทำนายผล
# ============================================================
with tab5:
    st.header("ทำนายผลจากลักษณะของเห็ด")
    st.warning(
        "⚠️ ระบบนี้จัดทำขึ้นเพื่อการศึกษาเท่านั้น **ห้ามใช้ตัดสินใจเก็บหรือรับประทานเห็ดจริงในธรรมชาติ** "
        "เพราะความผิดพลาดอาจเป็นอันตรายถึงชีวิต หากไม่แน่ใจในชนิดของเห็ด ควรปรึกษาผู้เชี่ยวชาญด้านพฤกษศาสตร์เห็ดโดยตรง"
    )

    st.subheader("กรอกลักษณะของเห็ด")
    user_values = {}
    cols = st.columns(3)
    for i, feature in enumerate(CATEGORY_OPTIONS.keys()):
        with cols[i % 3]:
            user_values[feature] = st.selectbox(
                FEATURE_LABELS_TH.get(feature, feature),
                CATEGORY_OPTIONS[feature],
                key=f"input_{feature}",
            )

    if st.button("🔍 ทำนายผล", type="primary", use_container_width=True):
        row = pd.Series(0, index=FEATURE_COLUMNS)
        for feature, value in user_values.items():
            col_name = f"{feature}_{value}"
            if col_name in row.index:
                row[col_name] = 1
        X_input = row.to_frame().T
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        st.divider()
        if pred == 1:
            st.error(f"### ⚠️ มีพิษ (Poisonous) — ความมั่นใจ {proba[1] * 100:.1f}%")
        else:
            st.success(f"### ✅ กินได้ (Edible) — ความมั่นใจ {proba[0] * 100:.1f}%")

st.divider()
st.caption("โมเดลและข้อมูลนี้จัดทำเพื่อการเรียนรู้ Machine Learning เท่านั้น · Dataset: UCI Mushroom Data Set")
