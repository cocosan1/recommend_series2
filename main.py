import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import openpyxl

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from pandas.core.common import random_state
# import lightgbm as lgb
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import mean_squared_error # モデル評価用(平均二乗誤差)
# from sklearn.metrics import r2_score # モデル評価用(決定係数)

#st
st.set_page_config(page_title='recommend_series')
st.markdown('### レコメンド アプリ')

#データ読み込み
df_zenkoku = pd.read_pickle('df_zenkoku7879.pickle')

st.markdown('###### １．分析対象得意先の絞込み')
#得意先範囲の設定
sales_max = st.number_input('分析対象得意先の売上上限を入力', key='sales_max', value=70000000)
sales_min = st.number_input('分析対象得意先の売上下限を入力', key='sales_min', value=2000000)

#salesによる絞込み
df_zenkoku2 = df_zenkoku.copy()
df_zenkoku2 = df_zenkoku2[(df_zenkoku2['sales'] >= sales_min) & (df_zenkoku2['sales'] <= sales_max)]
st.caption(f'対象得意先数: {len(df_zenkoku2)}')

img_yajirusi = Image.open('矢印.jpeg')
st.image(img_yajirusi, width=20)

#target選定用リスト
st.markdown('######  ２．target得意先の選択')
cust_text = st.text_input('得意先名の一部を入力 例）東京イ')

target_list = []
for cust_name in df_zenkoku2.index:
    if cust_text in cust_name:
        target_list.append(cust_name)

if target_list != '':
    # selectbox target ***
    target = st.selectbox(
        'target得意先:',
        target_list,   
    ) 
st.image(img_yajirusi, width=20)  

if target != '':

    #dfをアイテム列だけに絞る
    df_zenkoku3 = df_zenkoku2.drop(['sales', 'a_price'], axis=1)

    #相関係数 相関なし0-0.2 やや相関あり0.2-0.4 相関あり 0.4-0.7 強い相関 0.7-
    df_corr = df_zenkoku3.corr()

    #targetの売上を整理
    df_zenkoku_temp = df_zenkoku.drop(['sales', 'a_price'], axis=1)
    df_target_sales = df_zenkoku_temp.loc[target] #indexがシリーズ名/カラム名が店名のseries

    st.write(df_target_sales)