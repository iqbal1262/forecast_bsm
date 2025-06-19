# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt

st.set_page_config(page_title="Peramalan Pengeluaran BSM", layout="wide")
st.title("📊 Peramalan Pengeluaran Total Mingguan BSM")

# === SIDEBAR ===
st.sidebar.header("⚙️ Konfigurasi")

uploaded_file = st.sidebar.file_uploader("📂 Upload file Excel", type=["xlsx", "xls"])
jenis_data = st.sidebar.radio("🗭 Pilih Jenis BSM:", ["Kapal", "Non-Kapal"])

def_order = (2, 1, 2) if jenis_data == "Non-Kapal" else (1, 1, 1)
def_seasonal = (1, 1, 1, 52) if jenis_data == "Non-Kapal" else (2, 1, 1, 52)

st.sidebar.markdown("### Parameter SARIMA")
p = st.sidebar.number_input("p (AR)", 0, 10, def_order[0])
d = st.sidebar.number_input("d (Diff)", 0, 2, def_order[1])
q = st.sidebar.number_input("q (MA)", 0, 10, def_order[2])
P = st.sidebar.number_input("P (Seasonal AR)", 0, 10, def_seasonal[0])
D = st.sidebar.number_input("D (Seasonal Diff)", 0, 2, def_seasonal[1])
Q = st.sidebar.number_input("Q (Seasonal MA)", 0, 10, def_seasonal[2])
m = st.sidebar.number_input("m (Periode Musiman)", 1, 60, def_seasonal[3])

st.sidebar.markdown("### Pengaturan Evaluasi")
rolling_window = st.sidebar.number_input("Periode Rolling", 4, 52, 12)
forecast_horizon = st.sidebar.number_input("Periode Forecast", 4, 52, 12)

run_forecast = st.sidebar.button("🚀 Jalankan Forecast")

# === MAIN ===
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['BSM_CREATED_ON'] = pd.to_datetime(df['BSM_CREATED_ON'])

    if 'TOTAL' not in df.columns or 'KODEKAPAL' not in df.columns:
        st.error("❌ Kolom 'TOTAL' atau 'KODEKAPAL' tidak ditemukan di file.")
    else:
        df = df[df['KODEKAPAL'].notna()] if jenis_data == "Kapal" else df[df['KODEKAPAL'].isna()]

        df_aggr = df[['BSM_CREATED_ON', 'TOTAL']].copy()
        weekly = df_aggr.resample('W-MON', on='BSM_CREATED_ON').sum().reset_index()
        weekly['BSM_CREATED_ON'] = pd.to_datetime(weekly['BSM_CREATED_ON'])
        ts_original = weekly.set_index("BSM_CREATED_ON")["TOTAL"]

        st.subheader("📈 Total Pengeluaran Mingguan")
        st.plotly_chart(px.line(
            x=ts_original.index,
            y=ts_original.values,
            labels={'x': 'Tanggal', 'y': 'Total (Rp)'}
        ), use_container_width=True)

        # Deskriptif 1 baris dengan tanggal + styling kecil
        col_min, col_max, col_mean = st.columns(3)
        col_min.markdown(f"<div style='font-size:0.85rem;'>Minimum<br><strong>Rp {ts_original.min():,.0f}</strong><br>{ts_original.idxmin().strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
        col_max.markdown(f"<div style='font-size:0.85rem;'>Maximum<br><strong>Rp {ts_original.max():,.0f}</strong><br>{ts_original.idxmax().strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
        col_mean.markdown(f"<div style='font-size:0.85rem;'>Rata-rata<br><strong>Rp {ts_original.mean():,.0f}</strong></div>", unsafe_allow_html=True)

        if run_forecast:
            try:
                ts_boxcox, lam = boxcox(ts_original)

                model_insample = SARIMAX(ts_boxcox, order=(p,d,q), seasonal_order=(P,D,Q,m), trend='c',
                                         enforce_stationarity=False, enforce_invertibility=False)
                results_insample = model_insample.fit(disp=False)

                in_sample_pred = results_insample.fittedvalues
                in_sample_pred_original = pd.Series(inv_boxcox(in_sample_pred, lam), index=ts_original.index)
                actual_train_original = pd.Series(inv_boxcox(ts_boxcox, lam), index=ts_original.index)

                in_sample_pred_shifted = in_sample_pred_original.shift(-1)[2:-1]
                actual_train_aligned = actual_train_original[2:-1]

                start_point = len(ts_boxcox) - rolling_window
                predictions, actuals = [], []

                for i in range(rolling_window):
                    train_rolling = ts_boxcox[:start_point + i]
                    model = SARIMAX(train_rolling, order=(p,d,q), seasonal_order=(P,D,Q,m), trend='c',
                                    enforce_stationarity=False, enforce_invertibility=False)
                    results = model.fit(disp=False)
                    pred_bc = results.forecast(steps=1)
                    pred_real = inv_boxcox(pred_bc[0], lam)
                    predictions.append(pred_real)
                    actuals.append(ts_original.iloc[start_point + i])

                predictions_shifted = pd.Series(predictions).shift(-1)
                actuals_series = pd.Series(actuals)
                rolling_index = ts_original.index[start_point : start_point + len(actuals_series)]
                df_eval = pd.DataFrame({'actual': actuals_series.values, 'predicted': predictions_shifted.values},
                                       index=rolling_index).dropna()

                mae = mean_absolute_error(df_eval['actual'], df_eval['predicted'])
                rmse = sqrt(mean_squared_error(df_eval['actual'], df_eval['predicted']))
                mape = mean_absolute_percentage_error(df_eval['actual'], df_eval['predicted'])

                model_future = SARIMAX(ts_boxcox, order=(p,d,q), seasonal_order=(P,D,Q,m), trend='c',
                                       enforce_stationarity=False, enforce_invertibility=False)
                results_future = model_future.fit(disp=False)
                forecast_bc = results_future.forecast(steps=forecast_horizon)
                forecast_final = inv_boxcox(forecast_bc, lam)

                forecast_index = pd.date_range(start=ts_original.index[-1] + pd.Timedelta(weeks=1),
                                               periods=forecast_horizon, freq='W-MON')
                forecast_series = pd.Series(forecast_final, index=forecast_index)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_original.index, y=ts_original, name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=in_sample_pred_shifted.index, y=in_sample_pred_shifted,
                                         name='In-Sample Prediction', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df_eval.index, y=df_eval['predicted'],
                                         name='Rolling Forecast', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series,
                                         name='Forecast (Next)', line=dict(color='orange', dash='dot')))
                fig.update_layout(title='🔮 Timeline Prediksi',
                                  xaxis_title='Minggu',
                                  yaxis_title='Total Pengeluaran (Rp)',
                                  legend=dict(x=1.02, y=1),
                                  template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📏 Evaluasi Rolling Forecast")
                col1, col2, col3 = st.columns(3)
                col1.metric("MAPE", f"{mape:.2%}")
                col2.metric("MAE", f"Rp {mae:,.0f}")
                col3.metric("RMSE", f"Rp {rmse:,.0f}")

                st.markdown("### 🔍 Detail Forecast Mingguan")
                forecast_df = pd.DataFrame({"Tanggal": forecast_series.index.strftime('%Y-%m-%d'),
                                            "Forecast": forecast_series.values})

                df_eval['error_rolling'] = df_eval['actual'] - df_eval['predicted']
                error_in_sample = actual_train_aligned - in_sample_pred_shifted

                col_kde, col_table = st.columns([2, 1])
                with col_kde:
                    fig_error = go.Figure()
                    fig_error.add_trace(go.Histogram(x=df_eval['error_rolling'], name="Rolling Error", opacity=0.6))
                    fig_error.add_trace(go.Histogram(x=error_in_sample, name="In-sample Error", opacity=0.6))
                    fig_error.update_layout(title="Distribusi Error", barmode="overlay",
                                            template="simple_white", legend=dict(x=0.8, y=0.95))
                    st.plotly_chart(fig_error, use_container_width=True)
                with col_table:
                    st.dataframe(forecast_df.style.format({"Forecast": "Rp {:,.0f}"}), use_container_width=True, height=400)

            except Exception as e:
                st.error(f"❌ Gagal proses model: {e}")
else:
    st.info("⬅️ Upload file Excel di sidebar.")
