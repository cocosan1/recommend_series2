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

st.set_page_config(page_title='得意先分析')
st.markdown('## 得意先分析2')

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

#*****************************************************graphインスタンス
graph = Graph()

#*********************************************************基データ作成
#今期データ作成
df_nowbase = pd.DataFrame(index=df_now['得意先名'].unique())

items = []
for item in df_now['商品コード2'].unique():
    items.append(item)
    df_item = df_now[df_now['商品コード2']==item]
    s_cust = df_item.groupby('得意先名')['金額'].sum()
    df = pd.DataFrame(s_cust)
    df.rename(columns={'金額': item}, inplace=True)
    df_nowbase = df_nowbase.merge(df, left_index=True, right_index=True, how='left')
    df_nowbase = df_nowbase.fillna(0)

#item合計行作成
df_nowbase.loc['合計/item'] = df_nowbase.sum(axis=0)

#得意先合計行作成
df_nowbase['合計/cust'] = df_nowbase.sum(axis=1)

with st.expander('ベースデータ idx: 得意先名/ col: item df_nowbase'):
    st.dataframe(df_nowbase)
    st.write(df_nowbase.shape)

#前期データ作成
df_lastbase = pd.DataFrame(index=df_now['得意先名'].unique())

items = []
for item in df_last['商品コード2'].unique():
    items.append(item)
    df_item = df_last[df_last['商品コード2']==item]
    s_cust = df_item.groupby('得意先名')['金額'].sum()
    df = pd.DataFrame(s_cust)
    df.rename(columns={'金額': item}, inplace=True)
    df_lastbase = df_lastbase.merge(df, left_index=True, right_index=True, how='left')
    df_lastbase = df_lastbase.fillna(0)

#item合計行作成
df_lastbase.loc['合計/item'] = df_lastbase.sum(axis=0)

#得意先合計行作成
df_lastbase['合計/cust'] = df_lastbase.sum(axis=1)

with st.expander('ベースデータ idx: 得意先名/ col: item df_lastbase'):
    st.dataframe(df_lastbase)
    st.write(df_lastbase.shape)

s_now = df_nowbase['合計/cust'].sort_values(ascending=True)
s_now = s_now.drop(index='合計/item')
s_last = df_lastbase['合計/cust'].sort_values(ascending=True)
s_last = s_last.drop(index='合計/item')

#top30
col1, col2 = st.columns(2)
with col1:
    st.markdown('#### 今期 Top30')
    graph.make_bar_h_nonline(s_now[-30:], s_now.index[-30:], '今期', '売上金額', 700)

with col2:
    st.markdown('#### 前期 Top30')
    graph.make_bar_h_nonline(s_last[-30:], s_last.index[-30:], '前期', '売上金額', 700)