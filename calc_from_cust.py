import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
import openpyxl
import datetime

from sklearn.preprocessing import StandardScaler

from func_collection import Graph

st.set_page_config(page_title='ranking')
st.markdown('## 品番別分析')

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
df_now = DataFrame()
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
df_last = DataFrame()
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

#*****************************************************カテゴリー選択＋データ作成
st.markdown('### calc from cust')

cate_list = ['リビングチェア', 'ダイニングチェア', 'ダイニングテーブル', 'リビングテーブル', 'キャビネット類']
selected_cate = st.selectbox(
    '商品分類',
    cate_list,
    key='cl'
)
if selected_cate == 'リビングチェア':
    df_now2 = df_now[(df_now['商　品　名'].str.contains('ｿﾌｧ1P')) | 
                        (df_now['商　品　名'].str.contains('ｿﾌｧ2P')) |
                        (df_now['商　品　名'].str.contains('ｿﾌｧ2.5P')) | 
                        (df_now['商　品　名'].str.contains('ｿﾌｧ3P')) 
                        ] 
    
    df_last2 = df_last[(df_last['商　品　名'].str.contains('ｿﾌｧ1P')) | 
                        (df_last['商　品　名'].str.contains('ｿﾌｧ2P')) |
                        (df_last['商　品　名'].str.contains('ｿﾌｧ2.5P')) | 
                        (df_last['商　品　名'].str.contains('ｿﾌｧ3P')) 
                        ] 
        

    df_now2['品番'] = df_now2['商　品　名'].apply(lambda x: x.split(' ')[0])
    df_last2['品番'] = df_last2['商　品　名'].apply(lambda x: x.split(' ')[0])

    df_now2['数量'] = df_now2['数量'].fillna(0)
    df_last2['数量'] = df_last2['数量'].fillna(0)


else:
    df_now2 = df_now[df_now['商品分類名2']==selected_cate]
    df_last2 = df_last[df_last['商品分類名2']==selected_cate]

    df_now2['品番'] = df_now2['商　品　名'].apply(lambda x: x.split(' ')[0])
    df_last2['品番'] = df_last2['商　品　名'].apply(lambda x: x.split(' ')[0])

    df_now2['数量'] = df_now2['数量'].fillna(0)
    df_last2['数量'] = df_last2['数量'].fillna(0)

st.write(df_now2)
items = []
num_custs = []
cnts = []
for item in df_now2['商品コード2'].unique():
    items.append(item)
    df = df_now2[df_now2['商品コード2']==item]
    #得意先の数
    num_cust = df['得意先名'].nunique()
    #売れた商品の数量
    cnts = df['数量'].sum()





