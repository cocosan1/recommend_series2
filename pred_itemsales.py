import pandas as pd
import numpy as np
import streamlit as st
import openpyxl
import datetime

import func_collection as fc
from func_collection import Graph

st.set_page_config(page_title='アイテム売上予測')
st.markdown('## アイテム売上予測')

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

st.write(f'【得意先数】 {df_now_span["得意先名"].nunique()}')

def pred_itemsales():
    st.markdown('### アイテム概要/年換算')

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

    s_now2g = df_now2.groupby('品番')[selected_base].sum()
    s_last2g = df_last2.groupby('品番')[selected_base].sum()
    
    st.write(f'【今期数量】{s_now2g.sum(): .0f} 【前期数量】{s_last2g.sum(): .0f} \
             【対前年比】{s_now2g.sum() / s_last2g.sum():.2f}')
    
 
    
    #数量データ分布一覧/itemベース
    df_calc = fc.make_bunpu(df_now2, selected_base)
    
    st.markdown('##### 分布状況/年換算')

    with st.expander('df_calc', expanded=False):
        st.dataframe(df_calc)
    
    st.markdown('##### 分布状況中央値/年換算')
    with st.expander("df_calc 中央値", expanded=False):
        df_calc_median = df_calc[['得意先数', '中央値']]
        st.dataframe(df_calc_median)

    ######################################################### 品番選択
    hinban_list = sorted(list(df_now2['商品コード2'].unique()))
    hinbans = st.multiselect(
        '品番の選択/複数可',
        hinban_list,
        key='hinbans'
    )
    s_now = s_now2g[hinbans]

    st.write(df_calc.loc[hinbans])

    #前年の売上がない商品の対処
    for hinban in hinbans:
        if hinban in s_last2g.index:
            continue
        else:
            s_last2g[hinban] = 0

    s_last = s_last2g[hinbans]

#*****************************************************メイン
def main():
    # アプリケーション名と対応する関数のマッピング
    apps = {
        '-': None,
        'アイテム売上予測': pred_itemsales,

    }
    selected_app_name = st.sidebar.selectbox(label='分析項目の選択',
                                             options=list(apps.keys()))                                     

    if selected_app_name == '-':
        st.info('サイドバーから分析項目を選択してください')
        st.stop()

    # 選択されたアプリケーションを処理する関数を呼び出す
    render_func = apps[selected_app_name]
    render_func()

    link = '[home](https://cocosan1-hidastreamlit4-linkpage-7tmz81.streamlit.app/)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    st.sidebar.caption('homeに戻る') 

if __name__ == '__main__':
    main()
