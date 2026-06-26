import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib, json

st.set_page_config(
    page_title="Estimasi Harga Rumah Cidahu",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu,footer,header{visibility:hidden}
.stApp{background:#f6f8fc}
.block-container{padding-top:1.5rem}
section[data-testid='stSidebar']{
background:white;
border-right:1px solid #e5e7eb;
}
.hero{
background:white;
padding:32px;
border-radius:20px;
text-align:center;
box-shadow:0 8px 30px rgba(0,0,0,.05);
}
.result{
background:white;
padding:28px;
border-radius:18px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return (
        joblib.load("model_mlr.pkl"),
        joblib.load("model_rfr.pkl"),
        joblib.load("model_svr.pkl"),
    )

@st.cache_data
def load_meta():
    with open("metadata.json") as f:
        return json.load(f)

mlr, rfr, svr_bundle = load_models()
meta = load_meta()
svr_model = svr_bundle["model"]
scaler_X = svr_bundle["scaler_X"]
scaler_y = svr_bundle["scaler_y"]

FITUR = meta["fitur_terpilih"]

def prediksi(lb, lt):
    x = pd.DataFrame([[lb, lt]], columns=FITUR)
    p1 = float(mlr.predict(x)[0])
    p2 = float(rfr.predict(x)[0])
    p3 = float(scaler_y.inverse_transform(
        svr_model.predict(scaler_X.transform(x)).reshape(-1,1)
    )[0][0])
    return p1,p2,p3

with st.sidebar:
    st.title("🏡 Menu")
    page = st.radio(
        "Navigasi",
        [
            "🏠 Input Data",
            "💰 Hasil Estimasi",
            "📊 Analisis",
            "📚 Panduan"
        ]
    )

if "hasil" not in st.session_state:
    st.session_state.hasil=None

if page=="🏠 Input Data":

    st.markdown("""
    <div class='hero'>
    <h1>Estimasi Harga Rumah</h1>
    <p>Kecamatan Cidahu</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2=st.columns(2)

    with c1:
        lb=st.number_input("Luas Bangunan (m²)",30,1000,113)

    with c2:
        lt=st.number_input("Luas Tanah (m²)",30,10000,288)

    if st.button("Hitung Estimasi", use_container_width=True):
        st.session_state.hasil=(lb,lt,*prediksi(lb,lt))
        st.success("Estimasi berhasil dihitung. Buka menu Hasil Estimasi.")

elif page=="💰 Hasil Estimasi":

    if st.session_state.hasil is None:
        st.info("Isi data terlebih dahulu pada menu Input Data")

    else:
        lb,lt,p_mlr,p_rfr,p_svr=st.session_state.hasil

        st.markdown("<div class='result'>", unsafe_allow_html=True)
        st.metric(
            "Estimasi Harga (MLR)",
            f"Rp {p_mlr:,.0f}"
        )

        c1,c2,c3=st.columns(3)
        c1.metric("MLR",f"Rp {p_mlr:,.0f}")
        c2.metric("SVR",f"Rp {p_svr:,.0f}")
        c3.metric("RFR",f"Rp {p_rfr:,.0f}")

        st.markdown("</div>", unsafe_allow_html=True)

elif page=="📊 Analisis":

    if st.session_state.hasil:

        lb,lt,p_mlr,_,_=st.session_state.hasil

        x=np.linspace(50,2000,40)
        y=[]

        for i in x:
            y.append(prediksi(lb,i)[0])

        fig=go.Figure()
        fig.add_scatter(x=x,y=y)

        fig.update_layout(
            height=450,
            title="Simulasi Harga vs Luas Tanah"
        )

        st.plotly_chart(fig,use_container_width=True)

    else:
        st.info("Hitung estimasi terlebih dahulu")

else:

    st.markdown("""
### Cara Menggunakan
1. Buka Input Data
2. Isi luas bangunan & tanah
3. Hitung estimasi
4. Lihat hasil dan analisis
""")

