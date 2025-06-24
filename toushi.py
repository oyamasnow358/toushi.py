import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. テクニカル指標を計算する関数 (修正版) ---
def calculate_indicators(df):
    """日足データと週足データにテクニカル指標を追加する"""
    # --- 日足指標 ---
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean() # 中期トレンド用に50日線を追加
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_UPPER'] = df['SMA20'] + (rolling_std * 2)
    df['BB_LOWER'] = df['SMA20'] - (rolling_std * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rs = rs.fillna(0)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['High_60d'] = df['High'].rolling(window=60).max() # 利益確定目標用に過去60日の高値

    # --- 週足指標 (MACDの計算用) ---
    df_weekly = df.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    exp1_w = df_weekly['Close'].ewm(span=12, adjust=False).mean()
    exp2_w = df_weekly['Close'].ewm(span=26, adjust=False).mean()
    df_weekly['MACD'] = exp1_w - exp2_w

    return df.dropna(), df_weekly.dropna()

# --- 2. 『トリプル・フィルター・タイミング戦略』の判定ロジック関数 (大幅アップグレード版) ---
def analyze_signals(df, df_weekly):
    """テクニカル指標に基づいて売買シグナル、トレンド、戦略を分析する"""
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    latest_weekly = df_weekly.iloc[-1]

    signals = {'buy': [], 'sell': [], 'neutral': []}
    trends = {}
    advice = {'buy_targets': [], 'sell_targets': []}
    score = 0

    # --- トレンド分析 (長期・中期・短期) ---
    # 長期トレンド
    if latest['Close'] > latest['SMA200'] and latest_weekly['MACD'] > 0:
        trends['long'] = "🟢 上昇基調"
    elif latest['Close'] < latest['SMA200']:
        trends['long'] = "🔴 下降基調"
    else:
        trends['long'] = "🟡 中立/方向性不定"
    
    # 中期トレンド
    if latest['Close'] > latest['SMA50'] and latest['SMA50'] > df.iloc[-10]['SMA50']:
        trends['mid'] = "🟢 上昇"
    elif latest['Close'] < latest['SMA50']:
        trends['mid'] = "🔴 下降"
    else:
        trends['mid'] = "🟡 もみ合い"

    # 短期トレンド
    if latest['Close'] > latest['SMA20'] and latest['SMA20'] > df.iloc[-5]['SMA20']:
        trends['short'] = "🟢 上昇"
    elif latest['Close'] < latest['SMA20']:
        trends['short'] = "🔴 調整/下降"
    else:
        trends['short'] = "🟡 もみ合い"

    # --- シグナル分析 (スコアリング) ---
    if trends['long'] == "🟢 上昇基調":
        signals['buy'].append("✅ 長期トレンドが良好です。")
        if latest['Close'] <= latest['BB_LOWER']:
            signals['buy'].append("✅ Bバンド-2σにタッチ (売られすぎの可能性)")
            score += 1
        if 30 <= latest['RSI'] <= 45:
            signals['buy'].append(f"✅ RSIが{latest['RSI']:.1f}まで低下 (押し目の可能性)")
            score += 1
    else:
        signals['sell'].append("❌ 長期トレンドが下降・中立のため、買いは推奨されません。")

    if previous['MACD'] < previous['MACD_SIGNAL'] and latest['MACD'] > latest['MACD_SIGNAL']:
        signals['buy'].append("✅ MACDがゴールデンクロス (買いサイン)")
        score += 1
        if latest['Volume'] > latest['Volume_MA20'] * 1.5:
             signals['buy'].append("🔥 出来高急増を伴うクロス (信頼性UP)")
             score += 1
    
    # 売り・注意シグナル
    if latest['Close'] >= latest['BB_UPPER']:
        signals['sell'].append(f"⚠️ Bバンド+2σに到達 (過熱感)")
    if latest['RSI'] >= 70:
        signals['sell'].append(f"⚠️ RSIが{latest['RSI']:.1f} (買われすぎ)")
    if previous['MACD'] > previous['MACD_SIGNAL'] and latest['MACD'] < latest['MACD_SIGNAL']:
        signals['sell'].append("❌ MACDがデッドクロス (売りサイン)")
    if previous['Close'] > previous['SMA200'] and latest['Close'] < latest['SMA200']:
        signals['sell'].append("🚨【損切り警告】200日線を下抜け。長期トレンド転換の可能性。")

    # --- 売買戦略アドバイス ---
    # 買い目標
    if trends['long'] == "🟢 上昇基調":
        advice['buy_targets'].append(f"Bバンド -2σ: **{latest['BB_LOWER']:.2f}円**")
        advice['buy_targets'].append(f"20日移動平均線: **{latest['SMA20']:.2f}円**")
        advice['buy_targets'].append(f"50日移動平均線: **{latest['SMA50']:.2f}円**")
    else:
        advice['buy_targets'].append("長期トレンドが下降基調のため、押し目買いは推奨されません。")
        
    # 売り目標
    advice['sell_targets'].append(f"Bバンド +2σ: **{latest['BB_UPPER']:.2f}円**")
    advice['sell_targets'].append(f"直近60日高値: **{latest['High_60d']:.2f}円**")

    # --- 総合評価 ---
    final_score = 0
    if trends['long'] == "🟢 上昇基調":
        final_score = min(score, 4)
    
    star_rating = "★" * final_score + "☆" * (4 - final_score)
    
    if final_score >= 3:
        comment = "複数の買いシグナルが点灯しており、絶好の買い場が近い可能性があります。押し目買いのチャンスを伺いましょう。"
    elif final_score >= 1:
        comment = "長期トレンドが良好な中で、調整局面を迎えています。買いを検討できるタイミングです。"
    elif trends['long'] == "🔴 下降基調":
        comment = "長期トレンドが下降基調のため、積極的な買いはリスクが高いです。戻り売りに注意が必要です。"
    else:
        comment = "明確な方向性が出ていません。様子見が賢明かもしれません。"

    return {
        "star_rating": star_rating,
        "score": final_score,
        "comment": comment,
        "signals": signals,
        "trends": trends,
        "advice": advice
    }

# --- 3. チャートを描画する関数 (SMA50追加) ---
def plot_chart(df, ticker):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.1, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ローソク足'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='red', width=2), name='SMA 200 (長期)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='green', width=1.5), name='SMA 50 (中期)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1, dash='dash'), name='SMA 20 (短期)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color='rgba(100,100,100,0.5)', width=1), name='Bバンド上'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color='rgba(100,100,100,0.5)', width=1), fill='tonexty', fillcolor='rgba(100,100,100,0.1)', name='Bバンド下'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='出来高', marker_color='lightblue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA20'], line=dict(color='grey', width=1, dash='dash'), name='出来高MA20'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_shape(type='line', x0=df.index[0], y0=70, x1=df.index[-1], y1=70, line=dict(color='red', dash='dash'), row=3, col=1)
    fig.add_shape(type='line', x0=df.index[0], y0=30, x1=df.index[-1], y1=30, line=dict(color='green', dash='dash'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name='Signal', line=dict(color='orange')), row=4, col=1)
    fig.update_layout(title_text=f'{ticker} テクニカル分析チャート', height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="価格", row=1, col=1); fig.update_yaxes(title_text="出来高", row=2, col=1); fig.update_yaxes(title_text="RSI", row=3, col=1); fig.update_yaxes(title_text="MACD", row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)

# --- 4. Streamlitアプリのメイン部分 (表示をアップグレード) ---
st.set_page_config(layout="wide")
st.title('📈 進化版・タイミング戦略分析アプリ')
st.markdown("中期〜長期投資の売買タイミングを分析します。**トレンド・シグナル・戦略**を総合的に評価します。")

st.sidebar.header('分析設定')
ticker = st.sidebar.text_input('ティッカーシンボルを入力 (例: AAPL, 7203.T)', 'AAPL')
start_date = st.sidebar.date_input('分析開始日', pd.to_datetime('2022-01-01'))

if ticker:
    try:
        raw_data = yf.download(ticker, start=start_date, end=pd.to_datetime('today'))
        
        if raw_data.empty:
            st.error("ティッカーが見つからないか、データがありません。正しいティッカーを入力してください。(例: 米国株は `AAPL`、日本株は `7203.T`)")
        else:
            data = pd.DataFrame()
            data['Open'] = raw_data['Open']; data['High'] = raw_data['High']; data['Low'] = raw_data['Low']; data['Close'] = raw_data['Close']; data['Volume'] = raw_data['Volume']
            
            df, df_weekly = calculate_indicators(data)
            
            st.header(f'分析結果：{ticker}')
            analysis_result = analyze_signals(df, df_weekly)
            col1, col2 = st.columns([1.2, 1.8])

            with col1:
                st.subheader("📊 総合評価")
                st.metric(label="買いシグナル強度", value=analysis_result['star_rating'])

                # シグナル強度の解説
                score_descriptions = {
                    4: "★★★★: **絶好の買い場** - 複数の強力な買いシグナルが点灯。",
                    3: "★★★☆: **強い買いシグナル** - 買いの好機。エントリーを検討。",
                    2: "★★☆☆: **買い検討のサイン** - 押し目の可能性。監視を強化。",
                    1: "★☆☆☆: **弱い買いシグナル** - ポジティブな兆候あり。様子見。",
                    0: "☆☆☆☆: **シグナルなし** - 明確な買いサインなし。様子見が賢明。",
                }
                st.markdown(score_descriptions[analysis_result['score']])
                st.info(f"**コメント：** {analysis_result['comment']}")
                
                st.subheader("📈 トレンド分析")
                st.markdown(f"- **長期トレンド (200日線):** {analysis_result['trends']['long']}")
                st.markdown(f"- **中期トレンド (50日線):** {analysis_result['trends']['mid']}")
                st.markdown(f"- **短期トレンド (20日線):** {analysis_result['trends']['short']}")

                st.subheader("🎯 売買戦略アドバイス")
                with st.container(border=True):
                    st.markdown("**押し目買いの目標価格（目安）**")
                    for target in analysis_result['advice']['buy_targets']:
                        st.markdown(f" - {target}")
                with st.container(border=True):
                    st.markdown("**利益確定の目標価格（目安）**")
                    for target in analysis_result['advice']['sell_targets']:
                        st.markdown(f" - {target}")

                with st.expander("🔍 シグナルの詳細な根拠を見る"):
                    st.write("**買いシグナル:**")
                    if analysis_result['signals']['buy']:
                        for signal in analysis_result['signals']['buy']: st.markdown(f"  - {signal}")
                    else: st.markdown("  - なし")
                    
                    st.write("**売り・注意シグナル:**")
                    if analysis_result['signals']['sell']:
                        for signal in analysis_result['signals']['sell']: st.markdown(f"  - {signal}")
                    else: st.markdown("  - なし")

            with col2:
                plot_chart(df, ticker)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.error("ティッカーシンボルが正しいか、分析期間が十分かをご確認ください。")


st.sidebar.markdown("---")
st.sidebar.markdown("**免責事項**")
st.sidebar.info("本アプリは教育目的で作成されたものであり、特定の金融商品を推奨するものではありません。表示される価格はあくまでテクニカル分析に基づく目安であり、利益を保証するものではありません。投資の最終的な判断はご自身の責任で行ってください。")