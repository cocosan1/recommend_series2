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

st.set_page_config(page_title='分析2')
st.markdown('## 分析2')

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

#**********************************************************商品別概要
def overview():
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

        df_now2[selected_base] = df_now2[selected_base].fillna(0)
        df_last2[selected_base] = df_last2[selected_base].fillna(0)

        s_now2g = df_now2.groupby('品番')[selected_base].sum()
        s_last2g = df_last2.groupby('品番')[selected_base].sum()

    else:
        df_now2 = df_now[df_now['商品分類名2']==selected_cate]
        df_last2 = df_last[df_last['商品分類名2']==selected_cate]

        df_now2['品番'] = df_now2['商　品　名'].apply(lambda x: x.split(' ')[0])
        df_last2['品番'] = df_last2['商　品　名'].apply(lambda x: x.split(' ')[0])

        df_now2[selected_base] = df_now2[selected_base].fillna(0)
        df_last2[selected_base] = df_last2[selected_base].fillna(0)


        s_now2g = df_now2.groupby('品番')[selected_base].sum()
        s_last2g = df_last2.groupby('品番')[selected_base].sum()
    
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

        graph.make_bar_h_nonline(s_temp_now, s_temp_now.index, '今期', selected_base, 700)
        
        with st.expander('一覧', expanded=False):
            st.dataframe(s_now2g)

    with col2:
        st.markdown('#### 前期 Top30')
        #上位30取り出し
        s_temp_last = s_last2g.sort_values(ascending=False)
        s_temp_last = s_temp_last[0:30]
        #並べ替え
        s_temp_last.sort_values(ascending=True, inplace=True)
        graph.make_bar_h_nonline(s_temp_last, s_temp_last.index, '前期', selected_base, 700)
        
        with st.expander('一覧', expanded=False):
            st.dataframe(s_last2g) 
    
    #数量データ分布一覧/itemベース
    items = []
    cnts = []
    medis = []
    quan75s = []
    quan90s = []
    maxs = []
    for item in df_now2['品番'].unique():
        items.append(item)
        df_item = df_now2[df_now2['品番']==item]
        s_cust = df_item.groupby('得意先名')[selected_base].sum()
        cnt = s_cust.count()
        medi = s_cust.median()
        quan75 = round(s_cust.quantile(0.75), 1)
        quan90 = round(s_cust.quantile(0.9), 1)
        max_num = s_cust.max()

        cnts.append(cnt)
        medis.append(medi)
        quan75s.append(quan75)
        quan90s.append(quan90)
        maxs.append(max_num)
    df_calc = pd.DataFrame(list(zip(cnts, medis, quan75s, quan90s,maxs)), \
                    columns=['得意先数', '中央値', '第3四分位', '上位10%', '最大値'], index=items)
    
    st.markdown('#### 分布状況/アイテムベース')
    st.dataframe(df_calc)
    
    #####################################年間想定一覧
    data_span =  (df_now['受注日'].max() - df_now['受注日'].min()).days
    #days属性を使用してTimedeltaオブジェクトの日数を取得
    span_rate = 365 / data_span

    items2 = []
    mids = []
    top10s = []
    top1s = []
    for item in df_calc.index:
        items2.append(item)
        df = df_calc[df_calc.index == item]
        mid = round(df['中央値'] * span_rate, 1).values
        top10 = round(df['上位10%'] * span_rate, 1).values
        top1 = round(df['最大値'] * span_rate, 1).values

        mids.append(mid)
        top10s.append(top10)
        top1s.append(top1)

    df_pred = pd.DataFrame(list(zip(mids, top10s, top1s)), columns=['中央値', '上位10%', '最大値'], \
                            index=items2)
    st.markdown('#### 年間販売予測')
    st.dataframe(df_pred)

    ############################################################品番絞込み
    st.markdown('#### 品番検索')

    #品番検索
    part_word = st.text_input(
        '頭品番 例SN',
        key='whole_pw'
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
        df_now3 = df_now2[df_now2['品番'] == selected_item]
        df_last3 = df_last2[df_last2['品番'] == selected_item]

        s_now = df_now3.groupby('得意先名')[selected_base].sum()
        s_last = df_last3.groupby('得意先名')[selected_base].sum()

        s_now = s_now.sort_values()
        s_last = s_last.sort_values()

        #可視化
        st.markdown('##### 数量の分布/箱ひげ')
        st.write('得意先数')
        st.write(len(s_now))
        graph.make_box(s_now, s_last, ['今期', '前期'])

        #試算
        
        st.markdown('##### 年間販売予測')
        df_pred2 = df_pred[df_pred.index == selected_item]
        st.write(f'■ 中央値: {round(s_now.median()*span_rate)}')
        st.write(f'■ 上位90%: {round(s_now.quantile(0.9)*span_rate)}')
        st.write(f'■ 最大値: {round(s_now.max()*span_rate)}')
     
    
    fc.fukabori(df_now, df_now2, graph)


#*****************************************************数量ランキング/購入した得意先＋アイテム　スケール調整

def cnt_per_cust():
    st.markdown('### 数量ランキング/購入した得意先＋アイテム')
    st.write('スケール調整 1000万円')

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

    assumption = st.number_input(
        '想定年間売上',
        value=10000000,
        key='assumption'
    )

    #スケール調整用の比率算出
    #今期
    cust_dict_now = {}

    for cust in df_now['得意先名'].unique():
        sum_cust = df_now[df_now['得意先名']==cust]['金額'].sum()
        scale_rate = assumption / sum_cust
        cust_dict_now[cust] = scale_rate
    
    df_scale = pd.DataFrame(cust_dict_now, index= ['scale_rate']).T
    
    
    if selected_cate == 'リビングチェア':
        df_now2 = df_now[(df_now['商　品　名'].str.contains('ｿﾌｧ1P')) | 
                            (df_now['商　品　名'].str.contains('ｿﾌｧ2P')) |
                            (df_now['商　品　名'].str.contains('ｿﾌｧ2.5P')) | 
                            (df_now['商　品　名'].str.contains('ｿﾌｧ3P')) 
                            ] 

        df_now2['品番'] = df_now2['商　品　名'].apply(lambda x: x.split(' ')[0])
        df_now2[selected_base] = df_now2[selected_base].fillna(0)

    else:
        df_now2 = df_now[df_now['商品分類名2']==selected_cate]
        df_now2['品番'] = df_now2['商　品　名'].apply(lambda x: x.split(' ')[0])

        df_now2[selected_base] = df_now2[selected_base].fillna(0)
    
    #グループ化
    df_calc = df_now2.groupby(['商品コード2', '得意先名'], as_index=False)[selected_base].sum()


    df_calc =df_calc.merge(df_scale, left_on='得意先名', right_index=True, how='left')

    col_name = f'{selected_base}/scale'

    df_calc[col_name] = round(df_calc[selected_base] * df_calc['scale_rate'], 1)
    
    df_calc.sort_values(selected_base, ascending=False, inplace=True)
    st.dataframe(df_calc)

    fc.fukabori(df_now, df_now2, graph)

###########################################################################################展示分析
def tenji():
    st.markdown('### 展示分析')
    st.write('★ 更新は submiit ボタンで!')

    df_base = pd.read_excel(
        uploaded_file_now, sheet_name='受注委託移動在庫生産照会', usecols=[1, 3, 15, 16, 42, 50])  # index　ナンバー不要　index_col=0

    #index:item/col:cust
    df_zenkoku = fc.make_data_cust(df_base)

    with st.expander('df_zenkoku'):
        st.write(df_zenkoku)
    
    ########################得意先の選択
    st.markdown('####  得意先の選択')
    cust_text = st.text_input('得意先名の一部を入力 例）ケンポ')

    cust_list = []
    for cust_name in df_zenkoku.columns:
        if cust_text in cust_name:
            cust_list.append(cust_name)

    cust_list.insert(0, '--得意先を選択--')        

    cust_name = ''
    if cust_list != '':
        # selectbox target ***
        cust_name = st.selectbox(
            '得意先:',
            cust_list,   
        )
    
    if cust_name == '':
        st.info('得意先を選択してください')
        st.stop()


    #######################################展示品の分析
    st.markdown('#### 展示品の分析')
    cate_list = ['リビングチェア', 'ダイニングチェア', 'ダイニングテーブル', 'リビングテーブル', 'キャビネット類']
    selected_cate = st.selectbox(
        '商品分類',
        cate_list,
        key='tenji_cl'
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

        df_now2['金額'] = df_now2['金額'].fillna(0)
        df_last2['金額'] = df_last2['金額'].fillna(0)

        s_now2g = df_now2.groupby('品番')['金額'].sum()
        s_last2g = df_last2.groupby('品番')['金額'].sum()

    else:
        df_now2 = df_now[df_now['商品分類名2']==selected_cate]
        df_last2 = df_last[df_last['商品分類名2']==selected_cate]

        df_now2['品番'] = df_now2['商　品　名'].apply(lambda x: x.split(' ')[0])
        df_last2['品番'] = df_last2['商　品　名'].apply(lambda x: x.split(' ')[0])

        df_now2['金額'] = df_now2['金額'].fillna(0)
        df_last2['金額'] = df_last2['金額'].fillna(0)

    ###############################データスケール調整
     #数量データ分布一覧/itemベース

    #想定売上設定
    assumption = st.number_input(
        '想定年間売上',
        value=10000000,
        key='assumption'
    )

    items = []

    df_base = pd.DataFrame(index=df_now2['得意先名'].unique())
    for item in df_now2['品番'].unique():
        items.append(item)
        df_item = df_now2[df_now2['品番']==item]
        s_cust = df_item.groupby('得意先名')['金額'].sum()
        df = pd.DataFrame(s_cust)
        df.rename(columns={'金額': item}, inplace=True)
        df_base = df_base.merge(df, left_index=True, right_index=True, how='left')
    
    df_base = df_base.fillna(0)
    with st.expander('df_base', expanded=False):
        st.dataframe(df_base)

    #スケール調整用の比率算出
    cust_dict = {}
    #計算用に転置
    df_baset = df_base.T
    df_scale = pd.DataFrame()
    for cust in df_baset.columns:
        sum_cust = df_baset[cust].sum()
        scale_rate = assumption / sum_cust
        df_scale[cust] = round(df_baset[cust] * scale_rate)
    
    with st.expander('df_scale', expanded=False):
        st.dataframe(df_scale)
    
    #転置 index得意先/col品番
    df_scalet = df_scale.T
    df_scalet.fillna(0, inplace=True)


    #偏差値化
    df_deviation = pd.DataFrame(index=df_scalet.index)
    for item in df_scalet.columns:
        df = df_scalet[item]
        #販売した得意先だけで集計する
        df = df[df > 0]
        #標準偏差 母分散・母標準偏差ddof=0
        df_std = df.std(ddof=0)
        #平均値
        df_mean = df.mean()
        #偏差値
        df_deviation[item] = ((df_scalet[item] - df_mean) / df_std * 10 + 50)
        df_deviation.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        df_deviation = df_deviation.round(1)
    
    #得意先に絞ったseries
    s_cust_std = df_deviation.loc[cust_name]

    with st.expander('df_cust_std', expanded=False):
        st.dataframe(s_cust_std)
    
    #得意先の売上が0のitem行を削除
    df_non0 = df_scale[df_scale[cust_name] != 0]
    non0s = df_non0.index

    s_cust_std = s_cust_std.loc[non0s]

    s_cust_std.sort_values(ascending=False, inplace=True)
    #可視化
    st.markdown('##### 販売商品/偏差値')
    graph.make_bar(s_cust_std, s_cust_std.index)

    st.write('売上合計偏差値')
    #index得意先col売上のseries
    zenkoku_dict = {}
    for cust in df_zenkoku.columns:
        cust_sum = df_zenkoku[cust].sum()
        zenkoku_dict[cust] = cust_sum

    df_sales = pd.DataFrame(zenkoku_dict, index=['売上']).T

    #偏差値化
    df_deviation2 = pd.DataFrame(index=df_sales.index)

    #標準偏差 母分散・母標準偏差ddof=0
    s_std2 = df_sales.std(ddof=0)

    #平均値
    s_mean2 = df_sales.mean()

    #偏差値
    df_deviation2['偏差値'] = ((df_sales - s_mean2) / s_std2 * 10 + 50)
    df_deviation2.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df_deviation2 = df_deviation2.round(1)
    df_sales.sort_values('売上', ascending=False, inplace=True)
    df_deviation2.sort_values('偏差値', ascending=False, inplace=True)

    with st.expander('s_sales', expanded=False):
        st.write(df_sales)
    with st.expander('s_deviation2', expanded=False):
        st.write(df_deviation2)
    
    cust_dev = df_deviation2.loc[cust_name]
    st.write('■ 偏差値: 売上/全国')
    st.write(cust_dev)
    

 

    




#*****************************************************メイン
def main():
    # アプリケーション名と対応する関数のマッピング
    apps = {
        '-': None,
        'アイテム別概要': overview,
        '回転数/アイテム+店舗':cnt_per_cust,
        '展示分析': tenji
          
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






