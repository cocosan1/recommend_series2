import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import openpyxl
import datetime

from sklearn.decomposition import PCA #主成分分析
from sklearn.preprocessing import StandardScaler #標準化

import func_collection as fc
from func_collection import Graph

st.set_page_config(page_title='シリーズ別一覧')
st.markdown('## シリーズ別一覧')

#*******************************************************************************データ取り込み＋加工
@st.cache_data(ttl=datetime.timedelta(hours=1))
def make_data_now(file):
    df_now = pd.read_excel(
    file, sheet_name='受注委託移動在庫生産照会', \
        usecols=[1,2, 3, 8, 9, 10, 15, 16, 31, 42, 46, 50, 51]) #index　ナンバー不要　index_col=0

    df_now['伝票番号2'] = df_now['伝票番号'].apply(lambda x: x[:8])
    df_now['得意先CD2'] = df_now['得意先CD'].apply(lambda x:str(x)[0:5])
    df_now['商品コード2'] = df_now['商品コード'].apply(lambda x: x.split()[0]) #品番
    df_now['張地'] = df_now['商　品　名'].apply(lambda x: x.split()[2] if len(x.split()) >= 4 else '')

    # ***INT型への変更***
    df_now[['数量', '金額', '原価金額']] = df_now[['数量', '金額', '原価金額']].fillna(0).astype('int64')
    #fillna　０で空欄を埋める

    return df_now

@st.cache_data(ttl=datetime.timedelta(hours=1))
def make_data_last(file):
    df_last = pd.read_excel(
    file, sheet_name='受注委託移動在庫生産照会', \
        usecols=[1, 2, 3, 8, 9, 10, 15, 16, 31, 42, 46, 50, 51])
    
    df_last['伝票番号2'] = df_last['伝票番号'].apply(lambda x: x[:8])
    df_last['得意先CD2'] = df_last['得意先CD'].apply(lambda x:str(x)[0:5])
    df_last['商品コード2'] = df_last['商品コード'].apply(lambda x: x.split()[0]) #品番
    df_last['張地'] = df_last['商　品　名'].apply(lambda x: x.split()[2] if len(x.split()) >= 4 else '')

    # ***INT型への変更***
    df_last[['数量', '金額', '原価金額']] = df_last[['数量', '金額', '原価金額']].fillna(0).astype('int64')
    #fillna　０で空欄を埋める

    return df_last     


# ***ファイルアップロード 今期***
uploaded_file_now = st.sidebar.file_uploader('今期', type='xlsx', key='now')
df_now = pd.DataFrame()
if uploaded_file_now:
    df_now = make_data_now(uploaded_file_now)

    #データ範囲表示
    date_start =df_now['受注日'].min()
    date_end =df_now['受注日'].max()
    st.sidebar.caption(f'{date_start} - {date_end}')

else:
    st.info('今期のファイルを選択してください。')


# ***ファイルアップロード　前期***
uploaded_file_last = st.sidebar.file_uploader('前期', type='xlsx', key='last')
df_last = pd.DataFrame()
if uploaded_file_last:
    df_last = make_data_last(uploaded_file_last)

    #データ範囲表示
    date_start =df_last['受注日'].min()
    date_end =df_last['受注日'].max()
    st.sidebar.caption(f'{date_start} - {date_end}')
    
else:
    st.info('前期のファイルを選択してください。')
    st.stop()

################################得意先絞り込み
############################年換算
date_min = df_now['受注日'].min()
date_max = df_now['受注日'].max()
date_span = (date_max - date_min).days
date_rate = 365 / date_span

df_now_year = df_now.copy()
df_last_year = df_last.copy()

df_now_year['数量'] = df_now_year['数量'] * date_rate
df_last_year['数量'] = df_last_year['数量'] * date_rate
df_now_year['金額'] = df_now_year['金額'] * date_rate
df_last_year['金額'] = df_last_year['金額'] * date_rate

#小数点以下1桁
df_now_year['数量'] = df_now_year['数量'].round(1)
df_last_year['数量'] = df_last_year['数量'].round(1)
df_now_year['金額'] = df_now_year['金額'].round(1)
df_last_year['金額'] = df_last_year['金額'].round(1)

selected_span = st.selectbox(
    '売上レンジを指定',
    ['全得意先', '500万未満', '500万-1000万', '1000万-1500万', '1500万-2000万', '2000万以上'],
    key='range')

#得意先ごとに金額を合計
s_now = df_now_year.groupby('得意先名')['金額'].sum()

#得意先を売上レンジで絞る関数
def select_df(index_list):
    df_now_span = df_now_year[df_now_year['得意先名'].isin(index_list)]
    df_last_span = df_last_year[df_last_year['得意先名'].isin(index_list)]

    return df_now_span, df_last_span

df_now_span = pd.DataFrame()
df_last_span = pd.DataFrame()

if selected_span == '全得意先':
    df_now_span = df_now.copy()
    df_last_span = df_last.copy()

elif selected_span == '500万未満':
    s_now_selected = s_now[(s_now >= 0) & (s_now < 5000000)]
    index_list = s_now_selected.index

    df_now_span, df_last_span = select_df(index_list)
    
elif selected_span == '500万-1000万':
    s_now_selected = s_now[(s_now >= 5000000) & (s_now < 10000000)]
    index_list = s_now_selected.index
    
    df_now_span, df_last_span = select_df(index_list)

elif selected_span == '1000万-1500万':
    s_now_selected = s_now[(s_now >= 10000000) & (s_now < 15000000)]
    index_list = s_now_selected.index
    
    df_now_span, df_last_span = select_df(index_list)

elif selected_span == '1500万-2000万':
    s_now_selected = s_now[(s_now >= 15000000) & (s_now < 20000000)]
    index_list = s_now_selected.index
    
    df_now_span, df_last_span = select_df(index_list)

elif selected_span == '2000万以上':
    s_now_selected = s_now[(s_now >= 20000000)]
    index_list = s_now_selected.index
    
    df_now_span, df_last_span = select_df(index_list)


#*****************************************************graphインスタンス
graph = Graph()

#**********************************************************商品別概要 前年比

st.markdown('### アイテム概要')

selected_base = st.selectbox(
    '分析ベース選択',
    ['数量', '金額'],
    key='ov_sbase'
)

cate_list = ['リビングチェア', 'ダイニングチェア', 'ダイニングテーブル', 'リビングテーブル', 'キャビネット類']
selected_cate = st.selectbox(
    '商品分類',
    cate_list,
    key='cl'
)

#前処理
df_now2, df_last2 = fc.pre_processing(df_now_span, df_last_span, selected_base, selected_cate)

# series_name = st.selectbox('シリーズ名', df_now2['シリーズ名'].unique(), key='series_name')

# df_now2_selected = df_now2[df_now2['シリーズ名']==series_name]

# s_cust_product = df_now2_selected.groupby('得意先名')[selected_base].sum()

# st.write(s_cust_product)



#四分位処理

items = []
cnts = []
quan25s = []
medis = []
quan75s = []
quan90s = []
maxs = []
span2575s = []
den_cnts = []
den_maxs = []
den_rates = []

for item in df_now2['シリーズ名'].unique():
    items.append(item)
    df_item = df_now2[df_now2['シリーズ名']==item]
    s_cust = df_item.groupby('得意先名')[selected_base].sum()
    S_cust_den = df_item.groupby('得意先名')['伝票番号2'].nunique()
    
    cnt = s_cust.count()
    quan25 = round(s_cust.quantile(0.25), 1)
    medi = s_cust.median()
    quan75 = round(s_cust.quantile(0.75), 1)
    quan90 = round(s_cust.quantile(0.9), 1)
    max_num = s_cust.max()
    span2575 = (quan75 + quan25)/2
    den_cnt = df_item['伝票番号2'].nunique()
    den_max = S_cust_den.max()
    den_rate = round(den_cnt / cnt, 1)

    cnts.append(cnt)
    quan25s.append(quan25)
    medis.append(medi)
    quan75s.append(quan75)
    quan90s.append(quan90)
    maxs.append(max_num)
    span2575s.append(span2575)
    den_cnts.append(den_cnt)
    den_maxs.append(den_max)
    den_rates.append(den_rate)

df_calc = pd.DataFrame(list(zip(cnts, quan25s, medis, quan75s, quan90s, maxs, span2575s, den_cnts, \
                                den_maxs, den_rates)), \
                columns=['得意先数', '第2四分位', '中央値', '第3四分位', '上位10%', '最大値', 'span2575', \
                            '伝票数', '伝票数/max', '伝票数/得意先数'], index=items)

with st.expander('df_calc', expanded=False):
    st.write(df_calc)

selected_col = st.selectbox('項目選択', df_calc.columns, key='selected_col')

s_calc2 = df_calc[selected_col]
s_calc2 = s_calc2.sort_values(ascending=True)

with st.expander('df_calc2', expanded=False):
    st.write(s_calc2)

graph.make_bar_h_nonline(s_calc2, s_calc2.index, 'シリーズ名', 'ランキング', 1000)




# st.markdown('#### シリーズ分析')


# total_now = s_cust_product.sum()

# #可視化
# st.markdown('##### 数量の分布/箱ひげ')
# st.write('得意先数')
# st.write(len(s_cust_product))
# graph.make_box_now(s_cust_product,  '今期')