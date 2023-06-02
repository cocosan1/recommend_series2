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

items = []
num_custs = []
cnts = []
for item in df_now2['商品コード2'].unique():
    items.append(item)
    df = df_now2[df_now2['商品コード2']==item]
    #得意先の数
    num_cust = df['得意先名'].nunique()
    num_custs.append(num_cust)
    #売れた商品の数量
    cnt = df['数量'].sum()
    cnts.append(cnt)

df_calc = pd.DataFrame(list(zip(cnts, num_custs)), index=items, columns=['数量', '得意先数'])
df_calc['平均回転数/店舗'] = df_calc['数量'] / df_calc['得意先数']
st.write(df_calc)

#*******************************************アイテムの深堀
#******************データの絞込み
st.markdown('#### 品番検索: 売れ筋分析 塗色/張地/得意先')

#品番検索
part_word = st.text_input(
    '頭品番 例SN',
    key='pw'
)

if part_word != '':

    item_list = []
    for item in df_now2['品番'].unique():
        if part_word in item:
            item_list.append(item)

    selected_item = st.selectbox(
        '品番',
        item_list,
        key='sl'
    )
    df_select = df_now2[df_now2['品番'] == selected_item]

    #******************塗色分析
    st.markdown('##### 塗色別数量')
    s_color = df_select.groupby('塗色CD')['数量'].sum()

    s_color = s_color.sort_values(ascending=False)
    graph.make_bar(s_color, s_color.index)

    #*****************張地ランキング
    st.markdown('##### 張地別数量')
    selected_color = st.selectbox(
        '塗色を選択',
        s_color.index,
        key='sc'
    )

    df_color = df_select[df_select['塗色CD']==selected_color]
    s_fab = df_color.groupby('張地')['数量'].sum()

    s_fab = s_fab.sort_values(ascending=True)

    with st.expander('張地グラフ', expanded=False):
        graph.make_bar_h_nonline(s_fab, s_fab.index, '数量', '張地/数量順', 800)

    #******************塗色張地分析
    s_item = df_select.groupby('商　品　名')['数量'].sum()

    s_item = s_item.sort_values(ascending=True)

    with st.expander('売れ筋組み合わせ 塗色／張地', expanded=False):
        graph.make_bar_h_nonline(s_item, s_item.index, '数量', '売れ筋組み合わせ 塗色／張地', 800)

        st.dataframe(s_item)
    #********************購入している得意先
    s_sum = df_select.groupby('得意先名')['数量'].sum()

    s_sum = s_sum.sort_values(ascending=True)
    graph.make_bar_h_nonline(s_sum, s_sum.index, '数量', '得意先別数量', 800)

    #dataframe作成
    sales_list = []
    rep_list = []
    for cust in s_sum.index:
        #得意先の売上取得
        df_cust = df_now[df_now['得意先名'] == cust]
        sales_sum = df_cust['金額'].sum()
        sales_list.append(sales_sum)
        #得意先担当者名取得
        rep = df_cust.iloc[0]['営業担当者名']
        rep_list.append(rep)

    df_cust = pd.DataFrame(list(zip(s_sum, sales_list, rep_list)), index=s_sum.index, \
                    columns=['数量', '参考:売上/全商品合計', '担当者'])
    df_cust.sort_values('数量', ascending=False, inplace=True)

    with st.expander('一覧', expanded=False):
        st.dataframe(df_cust)
    
    #********************得意先別深堀
    st.markdown('##### 得意先を選択して明細を見る')
    selected_cust = st.selectbox(
        '得意先を選択',
        df_cust.index,
        key='scust')
    
    df_cust2 = df_select[df_select['得意先名']==selected_cust]
    s_cust = df_cust2.groupby('商　品　名')['数量'].sum()

    s_cust = s_cust.sort_values(ascending=True)

    with st.expander('選択した得意先の明細を見る: 組み合わせ 塗色／張地', expanded=False):
        graph.make_bar_h_nonline(s_cust, s_cust.index, '数量', '組み合わせ 塗色／張地', 500)
        st.dataframe(df_cust2)

    
    #アソシエーション分析へのリンク
    st.markdown('#### 同時に買われているアイテムを見る')
    df_concat = pd.DataFrame()
    for num in df_cust2['伝票番号2']:
        df = df_now[df_now['伝票番号2'] == num]
        df_concat = pd.concat([df_concat, df], join='outer')
    
    with st.expander('明細', expanded=False):
        col_list = ['得意先名', '商　品　名', '数量', '伝票番号2']
        st.table(df_concat[col_list])


    link = '[アソシエーション分析](https://cocosan1-association-fullhinban-cmy4cf.streamlit.app/)'
    st.markdown(link, unsafe_allow_html=True)






