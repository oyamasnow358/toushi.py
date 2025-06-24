import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import io

# --- 1. データ取得・準備関数 ---
@st.cache_data(ttl=86400) # 1日キャッシュ
def load_jpx_list():
    """JPXから東証上場銘柄一覧をダウンロードしてDataFrameを作成"""
    try:
        url = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/jpx_juni2024.xlsx'
        res = requests.get(url)
        res.raise_for_status()
        df = pd.read_excel(io.BytesIO(res.content))
        df = df[['コード', '銘柄名']]
        df = df.rename(columns={'コード': 'code', '銘柄名': 'name'})
        df = df[df['code'].apply(lambda x: isinstance(x, int))] # コードが数値でない行を除外
        df['code'] = df['code'].astype(str)
        df['display'] = df['code'] + ': ' + df['name']
        return df
    except Exception as e:
        st.error(f"銘柄リストの取得に失敗しました: {e}")
        return None

@st.cache_data(ttl=3600) # 1時間キャッシュ
def get_stock_data(ticker_symbol):
    """yfinanceから全期間の株価データを取得"""
    return yf.download(ticker_symbol, period="max", progress=False)

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker_symbol):
    """yfinanceからファンダメンタルズデータを取得"""
    info = yf.Ticker(ticker_symbol).info
    return {
        'pbr': info.get('priceToBook'),
        'per': info.get('trailingPE'),
        'company_name': info.get('longName', ticker_symbol)
    }

# --- 2. 分析ロジック関数 ---
def calculate_indicators(df):
    """テクニカル指標を計算"""
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_UPPER'] = df['SMA20'] + (rolling_std * 2)
    df['BB_LOWER'] = df['SMA20'] - (rolling_std * 2)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = (gain / loss).fillna(0)
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['High_60d'] = df['High'].rolling(window=60).max()
    df_weekly = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    exp1_w = df_weekly['Close'].ewm(span=12, adjust=False).mean()
    exp2_w = df_weekly['Close'].ewm(span=26, adjust=False).mean()
    df_weekly['MACD_W'] = exp1_w - exp2_w
    return df.dropna(subset=['SMA200']), df_weekly.dropna()

def analyze_signals(df, df_weekly):
    """テクニカル分析を実行"""
    if len(df) < 2 or len(df_weekly) < 1: return None 
    latest = df.iloc[-1]; previous = df.iloc[-2]; latest_weekly = df_weekly.iloc[-1]
    score = 0; signals = {'buy': [], 'sell': []}
    trends = {
        'long': "🟢 上昇" if latest['Close'] > latest['SMA200'] and latest_weekly['MACD_W'] > 0 else "🔴 下降" if latest['Close'] < latest['SMA200'] else "🟡 中立",
        'mid': "🟢 上昇" if latest['Close'] > latest['SMA50'] else "🔴 下降",
        'short': "🟢 上昇" if latest['Close'] > latest['SMA20'] else "🔴 調整"
    }
    if trends['long'] == "🟢 上昇":
        score += 1
        if latest['Close'] <= latest['BB_LOWER']: signals['buy'].append("✅ Bバンド-2σ (売られすぎ)"); score += 1
        if 30 <= latest['RSI'] <= 45: signals['buy'].append(f"✅ RSI {latest['RSI']:.1f} (押し目)"); score += 1
    if previous['MACD'] < previous['MACD_SIGNAL'] and latest['MACD'] > latest['MACD_SIGNAL']:
        signals['buy'].append("✅ MACDゴールデンクロス"); score += 1
    if latest['Close'] >= latest['BB_UPPER']: signals['sell'].append("⚠️ Bバンド+2σ (過熱感)")
    if latest['RSI'] >= 70: signals['sell'].append(f"⚠️ RSI {latest['RSI']:.1f} (買われすぎ)")
    final_score = min(score, 4) if trends['long'] == "🟢 上昇" else 0
    return {
        "score": final_score, "star_rating": "★" * final_score + "☆" * (4 - final_score),
        "comment": "絶好の買い場候補" if final_score >= 3 else "買い検討の好機" if final_score >= 1 else "下降トレンド、買いは危険" if trends['long'] == "🔴 下降" else "様子見推奨",
        "trends": trends, "advice": {
            "buy": [f"Bバンド -2σ: **{latest['BB_LOWER']:.2f}**", f"50日線: **{latest['SMA50']:.2f}**"] if trends['long'] == "🟢 上昇" else ["押し目買い非推奨"],
            "sell": [f"Bバンド +2σ: **{latest['BB_UPPER']:.2f}**", f"直近60日高値: **{latest['High_60d']:.2f}**"]
        }, "signals": signals
    }

def analyze_valuation(fundamentals):
    """ファンダメンタルズ分析を実行"""
    results = {}
    pbr = fundamentals.get('pbr')
    if isinstance(pbr, (int, float)):
        if pbr < 1.0: pbr_eval = "🟢 割安"
        elif pbr < 2.0: pbr_eval = "🟡 妥当圏"
        else: pbr_eval = "🔴 割高"
        results['pbr'] = f"**{pbr:.2f} 倍** ({pbr_eval})"
    else: results['pbr'] = "データなし"
    per = fundamentals.get('per')
    if isinstance(per, (int, float)):
        if per < 15.0: per_eval = "🟢 割安"
        elif per < 25.0: per_eval = "🟡 妥当圏"
        else: per_eval = "🔴 割高"
        results['per'] = f"**{per:.2f} 倍** ({per_eval})"
    else: results['per'] = "データなし"
    return results

# --- 3. UI・描画関数 ---
def plot_chart(df, ticker):
    """チャートを描画"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ローソク足'), row=1, col=1)
    for sma, color, name in [(200, 'red', '長期'), (50, 'green', '中期'), (20, 'orange', '短期')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA{sma}'], line=dict(color=color), name=f'SMA{sma} ({name})'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='出来高', marker_color='lightblue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.update_layout(title_text=f'分析チャート', height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 4. Streamlitアプリのメイン部分 ---
st.set_page_config(layout="wide"); st.title('📈 総合投資分析アプリ')
st.sidebar.header('銘柄選択')
jpx_df = load_jpx_list()
if jpx_df is not None:
    selected_stock = st.sidebar.selectbox('日本株を検索 (コード/会社名)', options=jpx_df['display'], index=None, placeholder="例: トヨタ or 7203")
    ticker_from_select = f"{selected_stock.split(':')[0]}.T" if selected_stock else None
else: ticker_from_select = None
ticker_from_input = st.sidebar.text_input('または、米国株等のティッカーを入力 (例: AAPL)', '')
ticker = ticker_from_select if ticker_from_select else ticker_from_input

if ticker:
    st.sidebar.header('チャート表示期間')
    period_options = {'3ヶ月': 90, '6ヶ月': 182, '1年': 365, '5年': 1825, '全期間': 99999}
    selected_period = st.sidebar.radio("期間", options=period_options.keys(), horizontal=True)

    raw_data = get_stock_data(ticker); fundamentals = get_fundamental_data(ticker)
    if raw_data.empty: st.error("データが見つかりません。ティッカーが正しいか確認してください。")
    else:
        analyzed_df, weekly_df = calculate_indicators(raw_data.copy())
        tech_analysis = analyze_signals(analyzed_df, weekly_df)
        val_analysis = analyze_valuation(fundamentals)
        
        st.header(f"{fundamentals.get('company_name', ticker)} の分析結果")
        col1, col2 = st.columns([1, 1.8])
        with col1:
            st.subheader("📊 テクニカル評価"); st.metric(label="買いシグナル強度", value=tech_analysis['star_rating'], help="長期トレンドが上昇基調の際の、買いタイミングの良さを示します。")
            st.info(f"**コメント:** {tech_analysis['comment']}")
            st.markdown(f"**トレンド:** 長期: {tech_analysis['trends']['long']} / 中期: {tech_analysis['trends']['mid']} / 短期: {tech_analysis['trends']['short']}")
            
            st.subheader("💰 ファンダメンタルズ評価");
            st.markdown(f"- **PBR (株価純資産倍率):** {val_analysis['pbr']}", help="1倍未満が割安の目安。企業の純資産に対して株価がどの程度かを示します。")
            st.markdown(f"- **PER (株価収益率):** {val_analysis['per']}", help="15倍未満が割安の目安。企業の利益に対して株価がどの程度かを示します。")
            
            st.subheader("🎯 売買戦略アドバイス")
            with st.container(border=True):
                st.markdown("**押し目買い目標 (目安)**"); [st.markdown(f" - {t}") for t in tech_analysis['advice']['buy']]
            with st.container(border=True):
                st.markdown("**利益確定目標 (目安)**"); [st.markdown(f" - {t}") for t in tech_analysis['advice']['sell']]
        with col2:
            display_df = analyzed_df[analyzed_df.index >= datetime.now() - timedelta(days=period_options[selected_period])]
            plot_chart(display_df, ticker)