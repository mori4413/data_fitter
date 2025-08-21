import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import StringIO

# ==== フィット関数定義 ====
def poly_fit(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(reversed(coeffs)))

def sine_fit(x, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * x + phi) + offset

def exp_fit(x, A, tau, offset):
    return A * np.exp(-x / tau) + offset

# ==== アプリ本体 ====
def run():
    st.title("📐 データフィッティングツール")

    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])
    if uploaded_file is None:
        st.info("ファイルをアップロードしてください。")
        return

    # === 安定して読み込むために StringIO へ変換 ===
    try:
        decoded = uploaded_file.read().decode("utf-8", errors="replace")
        string_io = StringIO(decoded)
        raw_lines = decoded.splitlines()
    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました: {e}")
        return

    preview_lines = [f"{i}: {line}" for i, line in enumerate(raw_lines[:30])]
    st.markdown("### 🧾 ファイルプレビュー（最初の30行）")
    st.selectbox("行プレビュー", options=preview_lines, index=0)

    col1, col2 = st.columns(2)
    with col1:
        header_row = st.number_input("📌 ヘッダー行番号", min_value=0, max_value=30, value=8)
    with col2:
        data_start = st.number_input("📌 データ開始行番号", min_value=0, max_value=30, value=9)

    # === 再読み込み ===
    try:
        string_io.seek(0)
        df = pd.read_csv(string_io, header=header_row,
                        skiprows=range(data_start) if data_start > header_row else None,
                        engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"CSV読み込み失敗: {e}")
        return

    st.markdown("### 📊 読み込まれたデータ")
    st.dataframe(df.head())

    colx, coly = st.columns(2)
    with colx:
        colx = st.selectbox("X軸に使用する列", df.columns)
    with coly:
        coly = st.selectbox("Y軸に使用する列", df.columns)

    x_data = df[colx].values
    y_data = df[coly].values

    st.markdown("### 🎯 フィッティング範囲の指定")
    min_x, max_x = float(np.nanmin(x_data)), float(np.nanmax(x_data))
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        region = st.slider("表示範囲", min_x, max_x, (min_x, max_x), step=(max_x - min_x) / 100)
    with col2:
        manual_start = st.number_input("解析開始位置", value=float(region[0]), format="%.4f")
    with col3:
        manual_end = st.number_input("解析終了位置", value=float(region[1]), format="%.4f")

    x_range = (manual_start, manual_end)
    idx = (x_data >= x_range[0]) & (x_data <= x_range[1])
    x_fit = x_data[idx]
    y_fit = y_data[idx]

    do_fit = st.toggle("📈 フィッティングを実行する", value=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        fit_func_name = st.selectbox("フィット関数を選択", ["多項式", "サイン波", "指数関数"])

    fit_func = None
    p0 = None

    if do_fit:
        with col2:
            if fit_func_name == "多項式":
                with col1:
                    degree = st.number_input("多項式の次数", min_value=1, max_value=10, value=2)

                fit_func = lambda x, *coeffs: poly_fit(x, *coeffs)
                p0 = []
                with col2:
                    for i in range(0, degree + 1, 2):
                        val = st.number_input(f"係数 a{i}", value=1.0, format="%.3f", key=f"a{i}")
                        p0.append(val)
                with col3:
                    for i in range(1, degree + 1, 2):
                        val = st.number_input(f"係数 a{i}", value=1.0, format="%.3f", key=f"a{i}")
                        p0.insert(i, val)

            elif fit_func_name == "サイン波":
                amp = st.number_input("振幅 A", value=1.0, format="%.3f")
                with col3:
                    period = st.number_input("周期", value=10.0, min_value=1.0, max_value=1000.0, step=1.0, format="%.2f")
                offset = st.number_input("オフセット", value=float(np.mean(y_fit)), format="%.3f")
                fit_func = sine_fit
                freq = 1.0 / period
                p0 = [amp, freq, 0.0, offset]

            elif fit_func_name == "指数関数":
                with col3:
                    amp = st.number_input("振幅 A", value=1.0, format="%.3f")
                    tau = st.number_input("時定数 τ", value=1.0, format="%.3f")
                offset = st.number_input("オフセット", value=float(np.mean(y_fit)), format="%.3f")
                fit_func = exp_fit
                p0 = [amp, tau, offset]

    # === グラフ描画 & フィット処理 ===
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_data, y_data, "x", label="Data", color="blue", alpha=0.4)
    ax.set_xlim(region)

    y_visible = y_data[(x_data >= region[0]) & (x_data <= region[1])]
    y_min = np.nanmin(y_visible)
    y_max = np.nanmax(y_visible)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(colx)
    ax.set_ylabel(coly)
    ax.set_title(f"Data plot & Fitting ({uploaded_file.name})")
    ax.grid(True)

    if do_fit:
        try:
            popt, _ = curve_fit(fit_func, x_fit, y_fit, p0=p0)
            y_pred = fit_func(x_fit, *popt)
            ax.plot(x_fit, y_pred, "-", label="Fit", color="red")

            residuals = y_fit - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r2 = 1 - (ss_res / ss_tot)

            ax.legend(loc="upper right")
            st.pyplot(fig)

            st.success("✅ フィッティング成功")
            st.markdown("### 📘 フィット係数と関数式")

            if fit_func_name == "多項式":
                terms = [f"{c:.3g}·x^{degree - i}" for i, c in enumerate(popt)]
                formula = " + ".join(terms).replace("x^0", "").replace("x^1", "x")
            elif fit_func_name == "サイン波":
                A, f, phi, offset = popt
                formula = f"{A:.3g}·sin(2π·{f:.3g}·x + {phi:.3g}) + {offset:.3g}"
            elif fit_func_name == "指数関数":
                A, tau, offset = popt
                formula = f"{A:.3g}·exp(-x / {tau:.3g}) + {offset:.3g}"

            st.latex(f"y = {formula}")
            for i, coeff in enumerate(popt):
                st.write(f"p[{i}] = {coeff:.6g}")
            st.write(f"**R² = {r2:.6f}**")
        except Exception as e:
            st.pyplot(fig)
            st.warning(f"⚠️ フィッティングに失敗しました: {str(e)}")
    else:
        st.pyplot(fig)
        st.info("📉 フィッティングはスキップされました。")

if __name__ == "__main__":
    run()