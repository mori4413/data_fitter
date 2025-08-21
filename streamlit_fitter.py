import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

    # 最初の数行を事前に確認
    try:
        preview_df = pd.read_csv(uploaded_file, header=None, nrows=10)
    except Exception as e:
        st.error(f"CSV読み込み失敗: {e}")
        return

    st.markdown("### 🧾 ファイルのプレビュー（先頭10行）")
    st.dataframe(preview_df)

    # ヘッダー行選択
    header_row = st.number_input("ヘッダー行の行番号を指定（0ベース）", min_value=0, max_value=9, value=0, step=1)
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(uploaded_file, header=header_row)
    except Exception as e:
        st.error(f"再読み込みに失敗しました: {e}")
        return

    st.markdown("### 📊 読み込まれたデータ")
    st.dataframe(df.head())

    colx = st.selectbox("X軸に使用する列", df.columns)
    coly = st.selectbox("Y軸に使用する列", df.columns)

    x_data = df[colx].values
    y_data = df[coly].values

    x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
    region = st.slider("フィットに使用するXの範囲", float(x_min), float(x_max), (float(x_min), float(x_max)))
    mask = (x_data >= region[0]) & (x_data <= region[1])
    x_fit = x_data[mask]
    y_fit = y_data[mask]

    fit_func_name = st.selectbox("フィット関数を選択", ["多項式", "サイン波", "指数関数"])
    fit_func = None
    p0 = None

    if fit_func_name == "多項式":
        degree = st.slider("多項式の次数", 1, 10, 2)
        fit_func = lambda x, *coeffs: poly_fit(x, *coeffs)
        p0 = [1.0] * (degree + 1)
    elif fit_func_name == "サイン波":
        fit_func = sine_fit
        p0 = [1.0, 0.01, 0, np.mean(y_fit)]
    elif fit_func_name == "指数関数":
        fit_func = exp_fit
        p0 = [1.0, 1.0, np.mean(y_fit)]

    try:
        popt, _ = curve_fit(fit_func, x_fit, y_fit, p0=p0)
        y_pred = fit_func(x_fit, *popt)
        residuals = y_fit - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r2 = 1 - (ss_res / ss_tot)

        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, label="元データ", color="gray", alpha=0.6)
        ax.plot(x_fit, y_fit, "o", label="フィット対象", color="blue")
        ax.plot(x_fit, y_pred, "-", label=f"フィット結果 ({fit_func_name})", color="red")
        ax.set_xlabel(colx)
        ax.set_ylabel(coly)
        ax.set_title("フィッティング結果")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.success("✅ フィッティング成功")
        st.markdown("### フィット係数")
        for i, coeff in enumerate(popt):
            st.write(f"p[{i}] = {coeff:.6g}")
        st.write(f"**R² = {r2:.6f}**")

    except Exception as e:
        st.error(f"フィッティングに失敗しました: {str(e)}")

if __name__ == "__main__":
    run()
