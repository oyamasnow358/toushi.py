import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

# --- 1. 初期設定 ---

# Streamlitページの基本設定
st.set_page_config(
    page_title="回帰分析 Webアプリ",
    page_icon="📊",
    layout="wide"
)

# --- 2. フォント設定 ---

### 修正箇所 ###
# Matplotlibの日本語フォント設定をより確実に行う関数に修正
def setup_japanese_font():
    """
    Matplotlib/Seabornで日本語を表示するためのフォントを設定します。
    IPAexゴシックフォントファイル（ipaexg.ttf）が同じディレクトリにあることを想定しています。
    """
    font_path = "ipaexg.ttf"
    
    if os.path.exists(font_path):
        # Matplotlibのフォントマネージャーにフォントを追加
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        
        # Matplotlibのデフォルト設定を変更
        plt.rcParams['font.family'] = font_prop.get_name()
        # マイナス記号の文字化け対策
        plt.rcParams['axes.unicode_minus'] = False
        
        # Seabornのスタイル設定にもフォントを適用
        # これにより、Seabornが生成するグラフ全体で日本語フォントが使われるようになります。
        sns.set_theme(style='whitegrid', font=font_prop.get_name())
    else:
        # フォントファイルが見つからない場合、サイドバーに警告を表示
        st.sidebar.warning("⚠️ 日本語フォントファイル（ipaexg.ttf）が見つかりません。グラフが文字化けします。")


# --- 3. UIコンポーネント関数 ---

def show_app_explanation():
    """初心者向けの回帰分析の説明を表示する"""
    with st.expander("🔍 回帰分析とは？（クリックで表示）", expanded=False):
        st.markdown("""
        ### **1. 「回帰分析」って何？**
        ある結果（例：売上）が、どんな要因（例：広告費）によって変化するのかを数式で解明する分析手法です。
        - **単回帰分析**: 1つの要因から結果を予測（例：「気温」から「アイスの売上」を予測）
        - **重回帰分析**: 複数の要因から結果を予測（例：「勉強時間」「睡眠時間」「食事」から「テストの点数」を予測）

        ### **2. 何がわかるの？**
        - **予測**: 新しいデータを使って、将来の結果を予測できます。
        - **要因分析**: どの要因が結果に最も強く影響しているかが分かります。
        - **関係性の可視化**: データ全体の傾向をグラフで直感的に理解できます。

        ### **3. このアプリで使われる指標**
        - **回帰係数**: 各要因が結果に与える影響の大きさ。プラスなら結果を増やし、マイナスなら減らす要因です。
        - **標準化回帰係数**: 各要因の単位（円、時間など）の影響を取り除いた係数。**この値の絶対値が大きいほど、結果への影響度が強い**と言えます。
        - **決定係数 (R²)**: モデルがデータ全体をどれだけ上手く説明できているかを示す指標（0〜1）。1に近いほど精度が高いと解釈できます。
        - **平均二乗誤差 (MSE)**: 予測値と実際の値のズレの平均。小さいほど予測精度が高いことを意味します。
        """)

def show_csv_template_section():
    """CSVテンプレートのダウンロードセクションを表示する"""
    st.markdown("#### 1. データを用意する")
    
    template_csv = """# このファイルは回帰分析用のデータテンプレートです。
# 「#」で始まる行はコメントとして扱われ、読み込み時に無視されます。
# 
# 【各列の説明】
# Y列: 目的変数 (予測したい値。例: 売上, テストの得点)
# X列: 説明変数 (予測に使う値。例: 広告費, 勉強時間)
#
# 以下にサンプルデータを示します。ご自身のデータに書き換えてお使いください。
目的変数,説明変数1,説明変数2,説明変数3
250,10,25,8
265,12,28,7
300,15,30,9
280,14,29,9
320,18,35,11
"""
    st.download_button(
        label="📥 CSVテンプレートをダウンロード",
        data=template_csv.encode('utf-8-sig'),
        file_name="regression_template.csv",
        mime="text/csv",
        help="分析に使用するデータ形式のサンプルです。"
    )

# --- 4. 分析ロジック関数 ---

def explain_correlation(corr_value):
    """相関係数の値を解釈する日本語の文字列を返す"""
    abs_corr = abs(corr_value)
    if abs_corr >= 0.7:
        return "かなり強い関係"
    elif abs_corr >= 0.4:
        return "やや強い関係"
    elif abs_corr >= 0.2:
        return "弱い関係"
    else:
        return "ほとんど関係なし"

def run_regression(df, target_var, feature_vars):
    """回帰分析を実行し、結果を辞書で返す"""
    try:
        # 欠損値を各列の平均値で補完
        X = df[feature_vars].fillna(df[feature_vars].mean())
        y = df[target_var].fillna(df[target_var].mean())

        # 線形回帰モデルの学習
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # モデル評価
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # 標準化回帰係数の計算
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model_scaled = LinearRegression()
        model_scaled.fit(X_scaled, y)
        
        # 結果をデータフレームにまとめる
        coef_df = pd.DataFrame({
            "変数名": feature_vars,
            "回帰係数": model.coef_,
            "標準化回帰係数": model_scaled.coef_,
            "目的変数との相関係数": [df[col].corr(df[target_var]) for col in feature_vars]
        })
        coef_df["相関の解釈"] = coef_df["目的変数との相関係数"].apply(explain_correlation)
        
        # 分析結果をまとめて返す
        return {
            "status": "success",
            "model": model, "X": X, "y": y, "y_pred": y_pred,
            "mse": mse, "r2": r2, "coef_df": coef_df, "intercept": model.intercept_
        }
    except Exception as e:
        return {"status": "error", "message": f"分析中にエラーが発生しました: {e}"}


# --- 5. 結果表示関数 ---

def display_analysis_results(results):
    """分析結果をタブ形式で表示する"""
    st.header("📊 分析結果", divider="rainbow")

    # 結果をタブで表示
    tab_summary, tab_visual, tab_data = st.tabs(["サマリー", "グラフで可視化", "使用データ"])

    # --- サマリータブ ---
    with tab_summary:
        st.subheader("📈 モデルの評価")
        col1, col2 = st.columns(2)
        col1.metric("決定係数 (R²)", f"{results['r2']:.4f}", help="モデルがデータの変動をどれだけ説明できているかを示す指標（1に近いほど良い）")
        col2.metric("平均二乗誤差 (MSE)", f"{results['mse']:.4f}", help="予測値と実測値の誤差の平均（0に近いほど良い）")
        
        st.subheader("🧮 回帰係数")
        st.write(f"**切片 (Intercept):** `{results['intercept']:.4f}`")
        st.dataframe(results['coef_df'].style.format('{:.4f}', subset=['回帰係数', '標準化回帰係数', '目的変数との相関係数'])
                                         .background_gradient(cmap='viridis', subset=['標準化回帰係数']))
        with st.expander("💡 表の見方"):
            st.markdown("""
            - **回帰係数**: その変数が1単位増加したときに、目的変数がどれだけ増減するかを示します。
            - **標準化回帰係数**: **影響の大きさ**を比較するための係数です。この値の絶対値が大きいほど、目的変数への影響力が強い変数だと解釈できます。
            - **目的変数との相関係数**: 個々の変数と目的変数の間の単純な関係の強さを示します。
            """)

    # --- 可視化タブ ---
    with tab_visual:
        # 実測値 vs 予測値プロット
        st.subheader("🎯 実測値 vs 予測値")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(results['y'], results['y_pred'], alpha=0.7, edgecolors='b')
        ax.plot([results['y'].min(), results['y'].max()], [results['y'].min(), results['y'].max()], 'r--', lw=2)
        ax.set_xlabel("実測値 (Actual)")
        ax.set_ylabel("予測値 (Predicted)")
        ax.set_title("実測値と予測値の比較")
        st.pyplot(fig)
        st.info("点が赤い点線に近いほど、モデルの予測が正確であることを示します。")

        # 残差プロット
        st.subheader("📉 残差プロット")
        residuals = results['y'] - results['y_pred']
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(results['y_pred'], residuals, alpha=0.7, edgecolors='g')
        ax.hlines(y=0, xmin=results['y_pred'].min(), xmax=results['y_pred'].max(), colors='red', linestyles='--')
        ax.set_xlabel("予測値 (Predicted)")
        ax.set_ylabel("残差 (Residuals)")
        ax.set_title("残差プロット")
        st.pyplot(fig)
        st.info("点が水平な赤い線の周りに均一に分布していれば、モデルは適切だと考えられます。")

        # 単回帰 or 重回帰のグラフ
        if len(results['X'].columns) == 1:
            st.subheader("📈 単回帰分析の可視化")
            feature_name = results['X'].columns[0]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(x=results['X'][feature_name], y=results['y'], ax=ax, line_kws={"color": "red"})
            ax.set_title(f"{feature_name} と {results['y'].name} の関係")
            st.pyplot(fig)
        else:
            st.subheader("🔗 説明変数間の相関ヒートマップ")
            corr_matrix = results['X'].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
            ax.set_title("説明変数間の相関")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
            st.info("変数の組み合わせで色が赤に近いほど「正の相関」、青に近いほど「負の相関」が強いことを示します。相関が強すぎる変数（例: 0.8以上）を両方モデルに入れると、多重共線性の問題が起きる可能性があります。")

    # --- データタブ ---
    with tab_data:
        st.subheader("📝 分析に使用したデータ")
        st.dataframe(pd.concat([results['y'], results['X']], axis=1).head(100))
        st.info("欠損値は各列の平均値で補完されています。")


# --- 6. メイン実行部 ---

def main():
    # 1. 初期設定の実行
    setup_japanese_font()

    # 2. アプリのタイトルと説明
    st.title("🚀 回帰分析 Webアプリ")
    st.write("CSVファイルをアップロードするだけで、簡単に重回帰分析が実行できるアプリです。")
    show_app_explanation()
    st.markdown("---")

    # 3. サイドバーの設定
    with st.sidebar:
        st.header("⚙️ 設定パネル")
        show_csv_template_section()

        st.markdown("#### 2. ファイルをアップロード")
        uploaded_file = st.file_uploader(
            "CSVファイルをここにドラッグ＆ドロップ",
            type=["csv"],
            help="ヘッダー行を含み、#で始まるコメント行は無視されます。"
        )
    
    # ファイルがアップロードされた後の処理
    if uploaded_file:
        try:
            # コメント行（#）を無視してCSVを読み込む
            df = pd.read_csv(uploaded_file, comment='#', encoding='utf-8-sig')
            
            # データが空でないかチェック
            if df.empty:
                st.error("❌ アップロードされたCSVファイルが空か、データ行がありません。")
                return

            st.success(f"✅ ファイル「{uploaded_file.name}」を正常に読み込みました。")
            
            with st.sidebar:
                st.markdown("#### 3. 変数を選択する")
                all_columns = df.columns.tolist()
                
                # 目的変数のデフォルト選択
                default_target_index = 0
                if "目的変数" in all_columns:
                    default_target_index = all_columns.index("目的変数")
                
                target_var = st.selectbox(
                    "目的変数 (Y)", all_columns, index=default_target_index,
                    help="予測したいターゲットとなる変数を選んでください。"
                )
                
                # 説明変数のデフォルト選択
                available_features = [col for col in all_columns if col != target_var]
                default_features = []
                # "説明変数"が含まれる列をデフォルトで選択
                for col in available_features:
                    if "説明変数" in col:
                        default_features.append(col)

                feature_vars = st.multiselect(
                    "説明変数 (X)", available_features, default=default_features,
                    help="予測に使用する変数を1つ以上選んでください。"
                )

                # 分析実行ボタン
                run_button = st.button("分析を実行", type="primary", use_container_width=True)

            if run_button:
                if not target_var or not feature_vars:
                    st.warning("⚠️ 目的変数と説明変数を1つ以上選択してください。")
                else:
                    with st.spinner("分析を実行中..."):
                        results = run_regression(df, target_var, feature_vars)
                        if results['status'] == 'success':
                            # 分析結果をセッションステートに保存
                            st.session_state['analysis_results'] = results
                        else:
                            st.error(f"❌ {results['message']}")
                            if 'analysis_results' in st.session_state:
                                del st.session_state['analysis_results'] # エラー時は古い結果を削除

        except Exception as e:
            st.error(f"❌ ファイルの読み込みまたは処理中にエラーが発生しました: {e}")
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
    
    # セッションステートに保存された結果があれば表示
    if 'analysis_results' in st.session_state:
        display_analysis_results(st.session_state['analysis_results'])

if __name__ == "__main__":
    main()