import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
import openpyxl
import datetime

from sklearn.preprocessing import StandardScaler

import func_collection as fc
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

#*****************************************************アイテム偏差値分析
def calc_deviation():
    #*******************************************上昇下降アイテムの抽出
    st.markdown('### アイテム上昇・下降分析/偏差値')

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

        s_now2g = df_now2.groupby('品番')['数量'].sum()
        s_last2g = df_last2.groupby('品番')['数量'].sum()

    else:
        df_now2 = df_now[df_now['商品分類名2']==selected_cate]
        df_last2 = df_last[df_last['商品分類名2']==selected_cate]

        df_now2['品番'] = df_now2['商　品　名'].apply(lambda x: x.split(' ')[0])
        df_last2['品番'] = df_last2['商　品　名'].apply(lambda x: x.split(' ')[0])

        df_now2['数量'] = df_now2['数量'].fillna(0)
        df_last2['数量'] = df_last2['数量'].fillna(0)


        s_now2g = df_now2.groupby('品番')['数量'].sum()
        s_last2g = df_last2.groupby('品番')['数量'].sum()
    
    st.write(f'【今期数量】{s_now2g.sum()} 【前期数量】{s_last2g.sum()} \
             【対前年比】{s_now2g.sum() / s_last2g.sum():.2f}')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### 今期 Top30')
        #上位30取り出し
        s_temp_now = s_now2g.sort_values(ascending=False)
        s_temp_now = s_temp_now[0:30]
        #並べ替え
        s_temp_now.sort_values(ascending=True, inplace=True)
        graph.make_bar_h_nonline(s_temp_now, s_temp_now.index, '今期', '今期/数量', 700)
        
        with st.expander('一覧', expanded=False):
            st.dataframe(s_now2g)

        #外れ値処理
        st.markdown('##### 外れ値処理')
        under_now = st.number_input('下限指定', key='unn', value=0)
        upper_now = st.number_input('上限指定', key='upn', value=100)

        s_now2g = s_now2g[(s_now2g >= under_now) & (s_now2g <= upper_now)]

        with st.expander('外れ値処理後', expanded=False):
            st.write(s_now2g)

    with col2:
        st.markdown('#### 前期 Top30')
        #上位30取り出し
        s_temp_last = s_last2g.sort_values(ascending=False)
        s_temp_last = s_temp_last[0:30]
        #並べ替え
        s_temp_last.sort_values(ascending=True, inplace=True)
        graph.make_bar_h_nonline(s_temp_last, s_temp_last.index, '前期', '前期/数量', 700)
        
        with st.expander('一覧', expanded=False):
            st.dataframe(s_last2g) 

        #外れ値処理
        st.markdown('##### 外れ値処理')
        under_last = st.number_input('下限指定', key='unl', value=0)
        upper_last = st.number_input('上限指定', key='upl', value=100)

        s_last2g = s_last2g[(s_last2g >= under_last) & (s_last2g <= upper_last)]
        with st.expander('外れ値処理後', expanded=False):
            st.write(s_last2g)


    #標準化
    #今期
    scaler = StandardScaler()
    s_now2gr = s_now2g.values.reshape(-1, 1) #.values忘れない #reshape(-1, 1)で縦配列に
    sd_now = scaler.fit_transform(s_now2gr)
    df_sdnow = pd.DataFrame(sd_now, columns=['今期'], index=s_now2g.index)

    #前期
    s_last2gr = s_last2g.values.reshape(-1, 1) #.values忘れない #reshape(-1, 1)で縦配列に
    sd_last = scaler.fit_transform(s_last2gr)
    df_sdlast = pd.DataFrame(sd_last, columns=['前期'], index=s_last2g.index)

    #merge
    df_m = df_sdnow.merge(df_sdlast, left_index=True, right_index=True, how='left')
    df_m = df_m.fillna(0)

    df_m['dev今期'] = df_m['今期'].apply(lambda x: (x*10)+50)
    df_m['dev前期'] = df_m['前期'].apply(lambda x: (x*10)+50)

    df_m['差異'] = df_m['dev今期'] - df_m['dev前期']
    df_m['比率'] = df_m['dev今期'] / df_m['dev前期']

    st.markdown('#### アイテム上昇・下降分析')
    #偏差値
    item_list = ['上昇アイテム', '下降アイテム']
    selected_item = st.selectbox(
        'アイテム選択',
        item_list,
        key='il'
    )

    #数量が平均より少ないアイテムの削除

    df_now2g2 = pd.DataFrame(s_now2g)
    df_last2g2 = pd.DataFrame(s_last2g)
    df_mval = df_now2g2.merge(df_last2g2, left_index=True, right_index=True, how='left')


    if selected_item == '上昇アイテム':
        df_up = df_m.sort_values(['比率', 'dev今期'], ascending=False)
        df_upm = df_up.merge(df_mval, left_index=True, right_index=True, how='left')
        df_upm.drop(['今期', '前期'], axis=1, inplace=True)
        df_upm = df_upm.rename(columns={'数量_x': '今期/数量', '数量_y': '前期/数量'})
        df_upm = df_upm[df_upm['比率'] >= 1.05]
        #ソート
        df_upm.sort_values('比率', ascending=True, inplace=True)
        #可視化
        graph.make_bar_h(df_upm['比率'], df_upm.index, '対前年比', '対前年比/偏差値/降順', 1, 1000)

        with st.expander('一覧', expanded=False):
            st.dataframe(df_upm)
    
    elif selected_item == '下降アイテム':
        df_down = df_m.sort_values('dev今期', ascending=False)
        df_down = df_m.sort_values('比率', ascending=True)
        df_downm = df_down.merge(df_mval, left_index=True, right_index=True, how='left')
        df_downm.drop(['今期', '前期'], axis=1, inplace=True)
        df_downm = df_downm.rename(columns={'数量_x': '今期/数量', '数量_y': '前期/数量'})
        df_downm = df_downm[df_downm['比率'] <= 0.95]

        #ソート
        df_downm.sort_values('比率', ascending=False, inplace=True)
        #可視化
        graph.make_bar_h(df_downm['比率'], df_downm.index, '対前年比', '対前年比/偏差値/昇順', 1, 1000)

        with st.expander('一覧', expanded=False):
            st.dataframe(df_downm)
    
    fc.fukabori(df_now, df_now2, graph)

#*****************************************************平均売上/購入した得意先
def mean_per_cust():
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

    fc.fukabori(df_now, df_now2, graph)

#*****************************************************数量ランキング/購入した得意先＋アイテム
def sum_per_cust():
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
    
    #グループ化
    df_calc = df_now2.groupby(['商品コード2', '得意先名'], as_index=False)['数量'].sum()

    st.dataframe(df_calc)

    fc.fukabori(df_now, df_now2, graph)



def main():
    # アプリケーション名と対応する関数のマッピング
    apps = {
        '-': None,
        '上昇・下降アイテム分析': calc_deviation,
        '平均/店舗': mean_per_cust,
        '合計/店舗': sum_per_cust
          
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






