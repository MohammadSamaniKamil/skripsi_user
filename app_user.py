"""
Aplikasi Estimasi Harga Rumah — Kecamatan Cidahu
Untuk pengguna umum: penjual, pembeli, agen properti
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib, json, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Estimasi Harga Rumah Cidahu",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 820px; }

  /* hero */
  .hero {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
    border-radius: 20px; padding: 36px 32px 28px 32px; margin-bottom: 28px;
    text-align: center; box-shadow: 0 8px 32px rgba(13,71,161,0.35);
  }
  .hero h1 { color: #fff; font-size: 28px; font-weight: 800; margin: 0 0 8px 0; line-height: 1.2; }
  .hero p  { color: #bbdefb; font-size: 14px; margin: 0; }
  .hero .badge-loc { display:inline-block; background:rgba(255,255,255,0.18); color:#e3f2fd;
    border-radius:20px; padding:4px 14px; font-size:12px; font-weight:600; margin-bottom:12px; }

  /* input card */
  .input-card {
    background: #fff; border-radius: 16px; padding: 28px 28px 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08); margin-bottom: 24px;
    border: 1px solid #e3e8f0;
  }
  .input-card h3 { color: #0d47a1; font-size: 16px; font-weight: 700; margin: 0 0 20px 0;
    padding-bottom: 10px; border-bottom: 2px solid #e3f2fd; }

  /* result main */
  .result-hero {
    background: linear-gradient(135deg, #1b5e20, #2e7d32, #388e3c);
    border-radius: 18px; padding: 32px; text-align: center;
    box-shadow: 0 6px 24px rgba(27,94,32,0.3); margin: 24px 0 16px;
  }
  .result-hero .label { color: #c8e6c9; font-size: 13px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
  .result-hero .price { color: #fff; font-size: 42px; font-weight: 800; line-height: 1.1; }
  .result-hero .sub   { color: #a5d6a7; font-size: 14px; margin-top: 8px; }
  .result-hero .range { color: #fff; font-size: 15px; margin-top: 12px;
    background:rgba(255,255,255,0.15); border-radius:10px; padding:8px 16px; display:inline-block; }

  /* model pills */
  .model-row { display:flex; gap:10px; margin:12px 0; flex-wrap:wrap; }
  .model-pill {
    flex:1; min-width:180px; border-radius:12px; padding:14px 16px; text-align:center;
    border: 1.5px solid transparent;
  }
  .pill-mlr { background:#dbeeff; border-color:#1565c0; }
  .pill-rfr { background:#d6f5dc; border-color:#2e7d32; }
  .pill-svr { background:#eedff7; border-color:#6a1b9a; }
  .model-pill .m-label { font-size:10px; font-weight:700; text-transform:uppercase;
    letter-spacing:0.5px; margin-bottom:4px; }
  .pill-mlr .m-label { color:#0a3880; }   /* lebih gelap agar kontras di bg biru muda */
  .pill-rfr .m-label { color:#145214; }   /* lebih gelap agar kontras di bg hijau muda */
  .pill-svr .m-label { color:#3a0070; }   /* lebih gelap agar kontras di bg ungu muda */
  .model-pill .m-price { font-size:18px; font-weight:800; }
  .pill-mlr .m-price { color:#0a3880; }
  .pill-rfr .m-price { color:#145214; }
  .pill-svr .m-price { color:#3a0070; }
  .model-pill .m-badge { font-size:10px; margin-top:3px; color:#444; font-weight:600; }  /* hapus opacity, pakai warna solid */
  .best-pill { box-shadow: 0 4px 12px rgba(13,71,161,0.2); transform: translateY(-2px); }

  /* tip box */
  .tip-box {
    background:#fff8e1; border-left:4px solid #f9a825; border-radius:8px;
    padding:12px 16px; font-size:13px; color:#5d4037; margin:12px 0;
  }
  .tip-box b { color:#e65100; }

  /* info box */
  .info-box {
    background:#e3f2fd; border-left:4px solid #1565c0; border-radius:8px;
    padding:12px 16px; font-size:13px; color:#0a2d6e; margin:12px 0;  /* warna teks lebih gelap */
    font-weight: 500;
  }

  /* comparison table */
  .comp-row { display:flex; align-items:center; padding:10px 0;
    border-bottom:1px solid #e8e8e8; }
  .comp-row:last-child { border-bottom: none; }
  .comp-label { flex:1; font-size:13px; color:#222; font-weight:500; }  /* #555 → #222 lebih gelap */
  .comp-bar-wrap { flex:2; background:#e0e0e0; border-radius:6px; height:10px; overflow:hidden; margin:0 12px; }  /* track lebih gelap dan tebal */
  .comp-bar { height:100%; border-radius:6px; }
  .comp-val { font-size:13px; font-weight:800; color:#111; width:80px; text-align:right; }  /* lebih bold dan gelap */

  /* section header */
  .sec-hdr {
    font-size:15px; font-weight:700; color:#0d47a1; margin:28px 0 14px;
    display:flex; align-items:center; gap:8px;
  }
  .sec-hdr::after { content:''; flex:1; height:2px; background:#e3f2fd; border-radius:2px; }

  /* pasaran card */
  .pasaran-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin:12px 0; }
  .pas-card { background:#fff; border:1.5px solid #e3e8f0; border-radius:12px;
    padding:14px; text-align:center; }
  .pas-card .pas-label { font-size:11px; color:#555; font-weight:700; text-transform:uppercase;
    letter-spacing:0.5px; margin-bottom:6px; }  /* #888 → #555 lebih terbaca */
  .pas-card .pas-val  { font-size:16px; font-weight:800; color:#0a3880; }  /* sedikit lebih gelap */
  .pas-card.highlight { background:#1565c0; border-color:#0a3880; }  /* background biru solid agar label putih terbaca */
  .pas-card.highlight .pas-label { color:#cce0ff; }  /* label putih di bg biru */
  .pas-card.highlight .pas-val { color:#ffffff; font-size:18px; }  /* harga putih tegas di bg biru */

  /* footer */
  .app-footer { text-align:center; color:#aaa; font-size:11px; margin-top:40px; padding-top:16px;
    border-top:1px solid #f0f0f0; }

  /* stSlider label */
  .stSlider label { font-weight: 600 !important; color: #333 !important; font-size: 14px !important; }

  /* number input */
  .stNumberInput label { font-weight: 600 !important; color: #333 !important; }

  /* radio */
  .stRadio label { font-size: 14px !important; }

  /* button */
  .stButton > button {
    background: linear-gradient(135deg, #0d47a1, #1565c0);
    color: white; border: none; border-radius: 12px;
    padding: 14px 28px; font-size: 16px; font-weight: 700;
    width: 100%; cursor: pointer; transition: all 0.2s;
    box-shadow: 0 4px 12px rgba(13,71,161,0.3);
  }
  .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(13,71,161,0.4); }
</style>
""", unsafe_allow_html=True)


# ─── Load ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    mlr = joblib.load("model_mlr.pkl")
    rfr = joblib.load("model_rfr.pkl")
    svr_b = joblib.load("model_svr.pkl")
    return mlr, rfr, svr_b

@st.cache_data
def load_meta():
    with open("metadata.json") as f:
        return json.load(f)

@st.cache_data
def load_df():
    df = pd.read_excel("dataUP.xlsx")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={
        "luas bangunan m2":"luas_bangunan","luas tanah m2":"luas_tanah",
        "aksesibilitas":"aksesibilitas_jalan","jumlah kamar tidur":"jumlah_kamar_tidur",
        "jumlah kamar mandi":"jumlah_kamar_mandi","jumlah lantai":"jumlah_lantai","harga (rp)":"harga_jual"
    })
    if not pd.api.types.is_numeric_dtype(df["harga_jual"]):
        df["harga_jual"] = df["harga_jual"].astype(str).str.strip().str.replace(".","",regex=False).str.replace(",",".",regex=False).astype(float)
    return df.dropna().reset_index(drop=True)

mlr_model, rfr_model, svr_bundle = load_models()
svr_model  = svr_bundle["model"]
scaler_X   = svr_bundle["scaler_X"]
scaler_y   = svr_bundle["scaler_y"]
meta = load_meta()
df   = load_df()
FITUR = meta["fitur_terpilih"]   # ["luas_bangunan", "luas_tanah"]

def prediksi(lb, lt):
    inp = pd.DataFrame([[lb, lt]], columns=FITUR)
    p_mlr = float(mlr_model.predict(inp)[0])
    p_rfr = float(rfr_model.predict(inp)[0])
    p_svr = float(scaler_y.inverse_transform(
        svr_model.predict(scaler_X.transform(inp)).reshape(-1,1))[0][0])
    return p_mlr, p_rfr, p_svr

def fmt_rp(x, short=False):
    if short:
        if x >= 1e9: return f"Rp {x/1e9:.2f} M"
        return f"Rp {x/1e6:.0f} Jt"
    return f"Rp {x:,.0f}"

def fmt_rp_range(low, high):
    return f"{fmt_rp(low, short=True)}  –  {fmt_rp(high, short=True)}"

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge-loc">📍 Kecamatan Cidahu, Kabupaten Sukabumi</div>
  <h1>🏡 Estimasi Harga Rumah</h1>
  <p>Masukkan spesifikasi rumah Anda dan dapatkan estimasi harga jual di Kecamatan Cidahu</p>
</div>
""", unsafe_allow_html=True)

# ─── INPUT FORM ────────────────────────────────────────────────────────────────

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📐 Luas Bangunan (m²)**")
        luas_bangunan = st.number_input(
            "Luas Bangunan",
            min_value=20, max_value=1000,
            value=113, step=5,
            help="Luas total area bangunan dalam meter persegi",
            label_visibility="collapsed"
        )
        
    with col2:
        st.markdown("**🌿 Luas Tanah (m²)**")
        luas_tanah = st.number_input(
            "Luas Tanah",
            min_value=20, max_value=10000,
            value=288, step=10,
            help="Luas total area tanah dalam meter persegi",
            label_visibility="collapsed"
        )
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Slider visual


    st.markdown("<br>", unsafe_allow_html=True)
    hitung = st.button("🔍  Hitung Estimasi Harga", use_container_width=True)

# ─── HASIL ─────────────────────────────────────────────────────────────────────
if not hitung:
    st.markdown("""
    <div style="
        background: #f0f4ff;
        border: 2px dashed #90a4d4;
        border-radius: 16px;
        padding: 36px 28px;
        text-align: center;
        margin-top: 20px;
    ">
        <div style="font-size: 48px; margin-bottom: 12px;">🏡</div>
        <div style="font-size: 18px; font-weight: 700; color: #0d47a1; margin-bottom: 8px;">
            Siap Menghitung Estimasi Harga
        </div>
        <div style="font-size: 14px; color: #5c6b8a;">
            Masukkan luas bangunan dan luas tanah di atas,<br>
            lalu tekan tombol <strong>Hitung Estimasi Harga</strong> untuk melihat hasilnya.
        </div>
    </div>
    """, unsafe_allow_html=True)

if hitung:
    p_mlr, p_rfr, p_svr = prediksi(luas_bangunan, luas_tanah)

    # Best model = MLR (R² tertinggi dari penelitian)
    best_pred = p_mlr
    # Rentang estimasi: gunakan MAE dari model terbaik sebagai margin
    mae_best  = meta["hasil_evaluasi"][0]["MAE"]   # MLR MAE
    low_est   = max(0, best_pred - mae_best)
    high_est  = best_pred + mae_best

    # ── Hasil utama
    st.markdown(f"""
    <div class="result-hero">
      <div class="label">Estimasi Harga Jual Terbaik</div>
      <div class="price">{fmt_rp(best_pred, short=True)}</div>
      <div class="sub">berdasarkan model Multiple Linear Regression (R² = 0.90)</div>
      <div class="range">📊 Kisaran wajar: {fmt_rp_range(low_est, high_est)}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Perbandingan 3 model
    st.markdown('<div class="sec-hdr">📊 Estimasi dari 3 Model</div>', unsafe_allow_html=True)

    models_info = [
        ("MLR", "Multiple Linear<br>Regression", p_mlr, "mlr", "R²=0.90 · Terbaik 🏆", True),
        ("SVR", "Support Vector<br>Regression",  p_svr, "svr", "R²=0.87", False),
        ("RFR", "Random Forest<br>Regression",   p_rfr, "rfr", "R²=0.78", False),
    ]

    cols = st.columns(3)
    for col_w, (abbr, label, price, css, badge, is_best) in zip(cols, models_info):
        with col_w:
            best_cls = "best-pill" if is_best else ""
            st.markdown(f"""
            <div class="model-pill pill-{css} {best_cls}">
              <div class="m-label">{label}</div>
              <div class="m-price">{fmt_rp(price, short=True)}</div>
              <div class="m-badge">{badge}</div>
            </div>
            """, unsafe_allow_html=True)


    # ── Gauge chart harga vs data
    st.markdown('<div class="sec-hdr">📍 Posisi Harga vs Data Pasar</div>', unsafe_allow_html=True)

    harga_min  = df["harga_jual"].min()
    harga_max  = df["harga_jual"].max()
    harga_q1   = df["harga_jual"].quantile(0.25)
    harga_med  = df["harga_jual"].median()
    harga_q3   = df["harga_jual"].quantile(0.75)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=best_pred,
        number={"prefix":"Rp ", "valueformat":",.0f", "font":{"size":20}},
        gauge={
            "axis": {"range":[harga_min, harga_max], "tickformat":".2s",
                     "tickprefix":"Rp ", "nticks":6},
            "bar":  {"color":"#1565c0", "thickness":0.3},
            "steps":[
                {"range":[harga_min, harga_q1],  "color":"#e3f2fd"},
                {"range":[harga_q1, harga_med],  "color":"#90caf9"},
                {"range":[harga_med, harga_q3],  "color":"#42a5f5"},
                {"range":[harga_q3, harga_max],  "color":"#1565c0"},
            ],
            "threshold":{"line":{"color":"#c62828","width":3},"thickness":0.85,"value":harga_med},
        },
        title={"text":"Posisi harga estimasi dalam rentang data pasar Kec. Cidahu",
               "font":{"size":13}},
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=60,b=10,l=30,r=30), paper_bgcolor="white")
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Label segmen
    if best_pred < harga_q1:
        segmen, segmen_color = "🟦 Segmen Bawah — harga lebih rendah dari 75% data pasar", "#0d47a1"
    elif best_pred < harga_med:
        segmen, segmen_color = "🟩 Segmen Menengah Bawah — di bawah rata-rata pasar", "#2e7d32"
    elif best_pred < harga_q3:
        segmen, segmen_color = "🟨 Segmen Menengah Atas — di atas rata-rata pasar", "#e65100"
    else:
        segmen, segmen_color = "🟥 Segmen Atas — termasuk 25% harga tertinggi di pasar", "#c62828"

    st.markdown(f"<div style='text-align:center;font-weight:700;color:{segmen_color};font-size:14px;margin-top:-8px'>{segmen}</div>", unsafe_allow_html=True)

    # ── Statistik pasar
    st.markdown('<div class="sec-hdr">🏘️ Referensi Harga Pasar Kec. Cidahu</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pasaran-grid">
      <div class="pas-card">
        <div class="pas-label">Harga Terendah</div>
        <div class="pas-val">{fmt_rp(harga_min, short=True)}</div>
      </div>
      <div class="pas-card">
        <div class="pas-label">Median Pasar</div>
        <div class="pas-val">{fmt_rp(harga_med, short=True)}</div>
      </div>
      <div class="pas-card">
        <div class="pas-label">Harga Tertinggi</div>
        <div class="pas-val">{fmt_rp(harga_max, short=True)}</div>
      </div>
      <div class="pas-card">
        <div class="pas-label">Batas Bawah (Q1)</div>
        <div class="pas-val">{fmt_rp(harga_q1, short=True)}</div>
      </div>
      <div class="pas-card highlight">
        <div class="pas-label">⭐ Estimasi Anda</div>
        <div class="pas-val">{fmt_rp(best_pred, short=True)}</div>
      </div>
      <div class="pas-card">
        <div class="pas-label">Batas Atas (Q3)</div>
        <div class="pas-val">{fmt_rp(harga_q3, short=True)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scatter posisi rumah di data
    st.markdown('<div class="sec-hdr">📈 Posisi Rumah Anda di Data Pasar</div>', unsafe_allow_html=True)

    fig_scatter = go.Figure()

    # Semua data
    fig_scatter.add_trace(go.Scatter(
        x=df["luas_tanah"], y=df["harga_jual"],
        mode="markers",
        marker=dict(color="#90caf9", size=8, opacity=0.6, line=dict(color="#1565c0", width=0.5)),
        name="Data pasar (110 rumah)",
        hovertemplate="Tanah: %{x}m²<br>Harga: Rp %{y:,.0f}<extra></extra>",
    ))

    # Rumah user
    fig_scatter.add_trace(go.Scatter(
        x=[luas_tanah], y=[best_pred],
        mode="markers+text",
        marker=dict(color="#c62828", size=18, symbol="star",
                    line=dict(color="white", width=2)),
        text=["  🏠 Rumah Anda"],
        textposition="middle right",
        textfont=dict(size=13, color="#c62828"),
        name="Estimasi Anda",
        hovertemplate=f"Tanah: {luas_tanah}m²<br>Estimasi: {fmt_rp(best_pred)}<extra></extra>",
    ))

    fig_scatter.update_layout(
        xaxis_title="Luas Tanah (m²)",
        yaxis_title="Harga Jual (Rp)",
        yaxis_tickformat=",.0f",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        margin=dict(t=20, b=40, l=60, r=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Faktor yang mempengaruhi harga
    st.markdown('<div class="sec-hdr">🔑 Faktor Penentu Harga Rumah di Cidahu</div>', unsafe_allow_html=True)

    fi = meta["feature_importance"]
    factors = [
        ("🌿 Luas Tanah",     fi["luas_tanah"]*100,    "#2e7d32", "Pengaruh terbesar — luas tanah menentukan 89% nilai rumah"),
        ("🏗️ Luas Bangunan",  fi["luas_bangunan"]*100, "#1565c0", "Pengaruh ke-2 — luas bangunan menentukan 7% nilai rumah"),
        ("🛏️ Kamar Tidur",   fi["jumlah_kamar_tidur"]*100, "#f9a825", "Pengaruh kecil (~1%)"),
        ("🏠 Jumlah Lantai",  fi["jumlah_lantai"]*100,  "#7b1fa2", "Pengaruh kecil (~1%)"),
        ("🚗 Aksesibilitas",  fi["aksesibilitas_jalan"]*100, "#e64a19", "Pengaruh sangat kecil (<1%)"),
        ("🚿 Kamar Mandi",    fi["jumlah_kamar_mandi"]*100, "#00695c", "Pengaruh sangat kecil (<1%)"),
    ]

    for label, pct, color, tooltip in factors:
        bar_w = int(pct)
        st.markdown(f"""
        <div class="comp-row" title="{tooltip}">
          <div class="comp-label">{label}</div>
          <div class="comp-bar-wrap">
            <div class="comp-bar" style="width:{min(bar_w,100)}%;background:{color}"></div>
          </div>
          <div class="comp-val">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


    # ── Simulasi harga berdasarkan luas tanah
    st.markdown('<div class="sec-hdr">📊 Simulasi Harga Berdasarkan Luas Tanah</div>', unsafe_allow_html=True)

    tanah_range = np.linspace(50, 2000, 60)
    harga_sim   = [prediksi(luas_bangunan, lt)[0] for lt in tanah_range]

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=tanah_range, y=harga_sim,
        mode="lines", name="Estimasi harga",
        line=dict(color="#1565c0", width=3),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.08)",
    ))
    fig_sim.add_trace(go.Scatter(
        x=[luas_tanah], y=[best_pred],
        mode="markers", name="Rumah Anda",
        marker=dict(color="#c62828", size=14, symbol="star"),
    ))
    fig_sim.update_layout(
        xaxis_title="Luas Tanah (m²)",
        yaxis_title="Estimasi Harga (Rp)",
        yaxis_tickformat=",.0f",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        margin=dict(t=20, b=40, l=60, r=20),
        title=dict(text=f"Estimasi harga untuk bangunan {luas_bangunan}m² dengan variasi luas tanah", font_size=13),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    # ── Panduan penggunaan
    st.markdown('<div class="sec-hdr">📌 Cara Membaca Estimasi Ini</div>', unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        **Untuk Penjual 🏷️**
        - Gunakan estimasi sebagai **harga dasar** negosiasi
        - Harga aktual bisa lebih tinggi jika lokasi strategis atau kondisi rumah sangat baik
        - Pertimbangkan juga kondisi pasar saat ini
        """)
    with col_g2:
        st.markdown("""
        **Untuk Pembeli 🛒**
        - Bandingkan harga penawaran dengan estimasi ini
        - Jika harga penawaran jauh di atas estimasi, negosiasikan
        - Lakukan survei lapangan untuk verifikasi kondisi fisik
        """)
