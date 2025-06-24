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
    
    # ▼▼▼【ここを修正】ボリンジャーバンドの計算を分割して堅牢にする ▼▼▼
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std() # 標準偏差を一度だけ計算
    df['BB_UPPER'] = df['SMA20'] + (rolling_std * 2)
    df['BB_LOWER'] = df['SMA20'] - (rolling_std * 2)
    # ▲▲▲【修正ここまで】▲▲▲

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # ゼロ除算を回避
    rs = gain / loss
    rs = rs.fillna(0) # lossが0の場合に発生するNaNを0で埋める
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 出来高移動平均
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # --- 週足指標 (MACDの計算用) ---
    df_weekly = df.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    exp1_w = df_weekly['Close'].ewm(span=12, adjust=False).mean()
    exp2_w = df_weekly['Close'].ewm(span=26, adjust=False).mean()
    df_weekly['MACD'] = exp1_w - exp2_w

    return df.dropna(), df_weekly.dropna()

# --- 2. 『トリプル・フィルター・タイミング戦略』の判定ロジック関数 ---
def analyze_signals(df, df_weekly):
    """テクニカル指標に基づいて売買シグナルを分析する"""
    # 最新のデータを取得
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    latest_weekly = df_weekly.iloc[-1]

    # スコアとコメントを初期化
    score = 0
    buy_signals = []
    sell_signals = []
    neutral_signals = []

    # --- フィルター1: 長期トレンド (方向性の確認) ---
    long_term_trend = "中立"
    if latest['Close'] > latest['SMA200'] and latest_weekly['MACD'] > 0:
        long_term_trend = "上昇基調"
        buy_signals.append("✅ **長期トレンド：** 株価が200日線より上にあり、週足MACDもプラス圏。")
    elif latest['Close'] < latest['SMA200']:
        long_term_trend = "下降基調"
        sell_signals.append("❌ **長期トレンド：** 株価が200日線を下回っており、注意が必要。")
    else:
        neutral_signals.append("🤔 **長期トレンド：** 方向性が明確ではありません。")

    # --- フィルター2: 中期タイミング (押し目・過熱感) ---
    # 買いのタイミングは長期トレンドが上昇基調の時のみ考慮
    if long_term_trend == "上昇基調":
        if latest['Close'] <= latest['BB_LOWER']:
            buy_signals.append("✅ **中期タイミング：** ボリンジャーバンド-2σにタッチ。売られすぎの可能性。")
            score += 1
        if 30 <= latest['RSI'] <= 45: # 押し目買いのRSI水準
             buy_signals.append(f"✅ **中期タイミング：** RSIが{latest['RSI']:.1f}まで低下。押し目の可能性。")
             score += 1

    # 過熱サイン (売り検討)
    if latest['Close'] >= latest['BB_UPPER']:
        sell_signals.append(f"⚠️ **過熱サイン：** ボリンジャーバンド+2σに到達。")
    if latest['RSI'] >= 70:
        sell_signals.append(f"⚠️ **過熱サイン：** RSIが{latest['RSI']:.1f}と高く、買われすぎを示唆。")

    # --- フィルター3: 短期エントリー/イグジット・トリガー ---
    # ゴールデンクロス (買いの引き金)
    if previous['MACD'] < previous['MACD_SIGNAL'] and latest['MACD'] > latest['MACD_SIGNAL']:
        buy_signals.append("✅ **短期トリガー：** MACDがゴールデンクロス。")
        score += 1
        if latest['Volume'] > latest['Volume_MA20'] * 1.5:
             buy_signals.append("🔥 **出来高急増：** ゴールデンクロスに出来高の増加が伴い、信頼性が高い。")
             score += 1 # 出来高の伴うクロスはボーナス点

    # デッドクロス (売りの引き金)
    if previous['MACD'] > previous['MACD_SIGNAL'] and latest['MACD'] < latest['MACD_SIGNAL']:
        sell_signals.append("❌ **短期トリガー：** MACDがデッドクロス。")

    # 損切りアラート
    if previous['Close'] > previous['SMA200'] and latest['Close'] < latest['SMA200']:
        sell_signals.append("🚨 **【損切り警告】** 200日線を下抜け。長期トレンド転換の可能性。")

    # --- 総合評価 ---
    final_score = 0
    if long_term_trend == "上昇基調":
        final_score = min(score, 4) # 最大スコアを4に制限
    
    star_rating = "★" * final_score + "☆" * (4 - final_score)
    
    if final_score >= 3:
        comment = "絶好の買い場が近づいている可能性があります。各指標がポジティブなサインを示しています。"
    elif final_score >= 1:
        comment = "買いを検討できる局面です。長期トレンドが良好な中で、押し目を形成しつつあります。"
    elif long_term_trend == "下降基調":
        comment = "長期トレンドが下降基調のため、積極的な買いは推奨されません。戻り売りに注意してください。"
    else:
        comment = "明確な売買サインは出ていません。様子見が賢明かもしれません。"

    return {
        "star_rating": star_rating,
        "comment": comment,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "neutral_signals": neutral_signals
    }

# --- 3. チャートを描画する関数 ---
def plot_chart(df, ticker):
    """Plotlyを使ってインタラクティブなチャートを作成する"""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.6, 0.1, 0.15, 0.15])

    # ローソク足、移動平均線、ボリンジャーバンド
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ローソク足'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='red', width=2), name='SMA 200'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1, dash='dash'), name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color='rgba(100,100,100,0.5)', width=1), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color='rgba(100,100,100,0.5)', width=1), fill='tonexty', fillcolor='rgba(100,100,100,0.1)', name='BB Lower'), row=1, col=1)

    # 出来高
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='出来高', marker_color='lightblue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA20'], line=dict(color='grey', width=1, dash='dash'), name='出来高MA20'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_shape(type='line', x0=df.index[0], y0=70, x1=df.index[-1], y1=70, line=dict(color='red', dash='dash'), row=3, col=1)
    fig.add_shape(type='line', x0=df.index[0], y0=30, x1=df.index[-1], y1=30, line=dict(color='green', dash='dash'), row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name='Signal', line=dict(color='orange')), row=4, col=1)

    fig.update_layout(
        title_text=f'{ticker} テクニカル分析チャート',
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="価格", row=1, col=1)
    fig.update_yaxes(title_text="出来高", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

# --- 4. Streamlitアプリのメイン部分 ---
st.set_page_config(layout="wide")
st.title('📈 トリプル・フィルター・タイミング戦略 アプリ')
st.markdown("中期〜長期投資の売買タイミングを分析します。良い企業（What）を、良いタイミング（When）で取引するためのツールです。")

# --- ユーザー入力 ---
st.sidebar.header('分析設定')
ticker = st.sidebar.text_input('ティッカーシンボルを入力 (例: AAPL, 7203.T)', 'AAPL')
start_date = st.sidebar.date_input('分析開始日', pd.to_datetime('2022-01-01'))

if ticker:
    try:
        # --- データ取得と計算 ---
        data = yf.download(ticker, start=start_date, end=pd.to_datetime('today'))
        if data.empty:
            st.error("ティッカーが見つからないか、データがありません。正しいティッカーを入力してください。(例: 米国株は `AAPL`、日本株は `7203.T`)")
        else:
            df, df_weekly = calculate_indicators(data)
            
            # --- 分析と結果表示 ---
            st.header(f'分析結果：{ticker}')
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("総合評価")
                analysis_result = analyze_signals(df, df_weekly)
                st.metric(label="買いシグナル強度", value=analysis_result['star_rating'])
                st.info(f"**コメント：** {analysis_result['comment']}")

                with st.expander("✅ 買いシグナルの詳細", expanded=True):
                    if analysis_result['buy_signals']:
                        for signal in analysis_result['buy_signals']: st.write(signal)
                    else: st.write("現在、明確な買いシグナルはありません。")

                with st.expander("⚠️ 売り・注意シグナルの詳細", expanded=True):
                    if analysis_result['sell_signals']:
                        for signal in analysis_result['sell_signals']: st.write(signal)
                    else: st.write("現在、明確な売り・注意シグナルはありません。")
                
                with st.expander("🤔 中立的な観察", expanded=False):
                    if analysis_result['neutral_signals']:
                        for signal in analysis_result['neutral_signals']: st.write(signal)
                    else: st.write("-")

            with col2:
                plot_chart(df, ticker)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**免責事項**")
st.sidebar.info("本アプリは教育目的で作成されたものであり、特定の金融商品を推奨するものではありません。投資の最終的な判断はご自身の責任で行ってください。")