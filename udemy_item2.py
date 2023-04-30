import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import openpyxl

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#*********************************************ユーザーベースの協調フィルタリング

#st
st.set_page_config(page_title='recommend_series')
st.markdown('### レコメンド ユーザーBアプリ/専門店2')

#データ読み込み
df_zenkoku = pd.read_pickle('df_zenkoku79v2.pickle')

st.markdown('#### １．分析対象得意先の絞込み')
#得意先範囲の設定
sales_max = st.number_input('分析対象得意先の売上上限を入力', key='sales_max', value=70000000)
sales_min = st.number_input('分析対象得意先の売上下限を入力', key='sales_min', value=2000000)

#salesによる絞込み
#コピーと転置
df_zenkoku2 = df_zenkoku.copy().T
df_zenkoku2 = df_zenkoku2[(df_zenkoku2['sales'] >= sales_min) & (df_zenkoku2['sales'] <= sales_max)]
st.caption(f'対象得意先数: {len(df_zenkoku2)}')

img_yajirusi = Image.open('矢印.jpeg')
st.image(img_yajirusi, width=20)

#target選定用リスト
st.markdown('####  ２．target得意先の選択')
cust_text = st.text_input('得意先名の一部を入力 例）ケンポ')

target_list = []
for cust_name in df_zenkoku2.index:
    if cust_text in cust_name:
        target_list.append(cust_name)

target_list.insert(0, '--得意先を選択--')        

if target_list != '':
    # selectbox target ***
    target = st.selectbox(
        'target得意先:',
        target_list,   
    ) 
st.image(img_yajirusi, width=20)  

if target != '--得意先を選択--':

    #疎行列（ほとんどが0の行列）アイテム/ユーザー行列
    #dfをアイテム列だけに絞る
    df_zenkoku3 = df_zenkoku2.drop(['sales'], axis=1)
    df_zenkoku3 = df_zenkoku3.fillna(0)
    # df_zenkoku3t = df_zenkoku3.T

    with st.expander('得意先別/アイテム別売上'):
        st.write(df_zenkoku3)
        st.caption('df_zenkoku3')

    #targetの売上を整理
    df_target_sales = df_zenkoku3.loc[target] #indexがシリーズ名/カラム名が店名のseries

    with st.expander('target得意先/シリーズ別売上/年間', expanded=False):
        st.write(df_target_sales)
        st.caption('df_target_sales')

    st.markdown('####  ３．展示品の選択')

    cate_list = list(df_target_sales.index)

    tenji_list = []
    # チェックボックスを表示し、ユーザーが選択したオプションを取得
    for cate in cate_list:
        if st.checkbox(cate):
            tenji_list.append(cate)


    st.image(img_yajirusi, width=20) 

    st.markdown('####  ４．動きのよくない展示品の抽出')

    # ***ファイルアップロード 今期***
    uploaded_file_now = st.file_uploader('今期受注ファイル', type='xlsx', key='now')

    if uploaded_file_now:
        df_now = pd.read_excel(
            uploaded_file_now, sheet_name='受注委託移動在庫生産照会', usecols=[3, 15, 42, 50])  # index　ナンバー不要　index_col=0
    else:
        st.info('今期のファイルを選択してください。')
        st.stop() 

    
    #LD分類
    df_now_target = df_now[df_now['得意先名']==target]
    cate_list = []
    for cate in df_now_target['商品分類名2']:
        if cate in ['ダイニングテーブル', 'ダイニングチェア', 'ベンチ']:
            cate_list.append('d')
        elif cate in ['リビングチェア', 'クッション', 'リビングテーブル']:
            cate_list.append('l')
        else:
            cate_list.append('none') 
    #分類列を追加
    df_now_target['category'] = cate_list 
    #noneを削除
    df_now_target = df_now_target[df_now_target['category']!='none']
    #シリーズ名＋LD分類
    df_now_target['series2'] = df_now_target['category'] + '_' + df_now_target['シリーズ名']

    with st.expander('target得意先に絞った表', expanded=False):
        st.write(df_now_target)
        st.caption('df_now_target') 

    #必要なカラムに絞る
    df_now_target = df_now_target[['得意先名', '金額', 'series2']] 

    #商品リストの絞込み　手作業
    selected_list = ['d_ALMO (ｱﾙﾓ)',
                    'd_AWASE',
                    'd_BAGUETTE LB(ﾊﾞｹｯﾄｴﾙﾋﾞｰ)',
                    'd_CHIGUSA(ﾁｸﾞｻ）',
                    'd_COBRINA',
                    'd_HIDA',
                    'd_Kinoe',
                    'd_L-CHAIR',
                    'd_Northern Forest',
                    'd_PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）',
                    'd_SEOTO',
                    'd_SEOTO-EX',
                    'd_TUGUMI',
                    'd_VIOLA (ｳﾞｨｵﾗ)',
                    'd_YURURI',
                    'd_nae',
                    'd_tsubura',
                    'd_クレセント',
                    'd_侭 JIN',
                    'd_侭SUGI',
                    'd_円空',
                    'd_杜の詩',
                    'd_森のことば',
                    'd_森のことば ウォルナット',
                    'd_森のことばIBUKI',
                    'd_穂高',
                    'd_風のうた',
                    'd_ﾆｭｰﾏｯｷﾝﾚｲ',
                    'l_ALMO (ｱﾙﾓ)',
                    'l_AWASE',
                    'l_CHIGUSA(ﾁｸﾞｻ）',
                    'l_COBRINA',
                    'l_Kinoe',
                    'l_Northern Forest',
                    'l_PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）',
                    'l_SEION 静穏',
                    'l_SEOTO',
                    'l_SEOTO-EX',
                    'l_TUGUMI',
                    'l_VIOLA (ｳﾞｨｵﾗ)',
                    'l_YURURI',
                    'l_nae',
                    'l_杜の詩',
                    'l_森のことば',
                    'l_森のことば ウォルナット',
                    'l_森のことばIBUKI',
                    'l_穂高',
                    'l_風のうた',
                    'l_ｽﾀﾝﾀﾞｰﾄﾞｺﾚｸｼｮﾝ',
                    'l_ﾆｭｰﾏｯｷﾝﾚｲ'
                    ]
    
    #商品毎に集計
    df_calc = pd.DataFrame(index=selected_list)
    
    val_list = []
    for cate in selected_list:
        df_cust2 = df_now_target[df_now_target['series2']==cate]
        if len(df_cust2) == 0:
            val_list.append(0)
        else:
            val = df_cust2['金額'].sum()
            val_list.append(val)
    df_temp = pd.DataFrame(val_list, columns=[f'{target}_now'], index=selected_list)
    df_calc = df_calc.merge(df_temp, left_index=True, right_index=True, how='outer') 

    df_calc.loc['sales'] = df_calc.sum()
    
    with st.expander('index:商品名/col:得意先名', expanded=False):
        st.write(df_calc)
        st.caption('df_calc') 


    #展示品の売り上げ上限を入力
    max_line = st.number_input('いくら以下を抽出するか', key='max_line', value=100000)
    #展示品に絞込み
    df_cold_sales = df_calc.loc[tenji_list]
    #売上下限以下のdfを作成
    df_cold_sales = df_cold_sales[df_cold_sales[f'{target}_now'] <= max_line]
    df_cold_sales = df_cold_sales.sort_values(f'{target}_now', ascending=False)
    
    st.markdown('##### 動いていない展示品')
    st.caption('df_cold_sales')
    st.write(df_cold_sales) 
    
    # with st.expander('得意先別/シリーズLD別/売上表', expanded=False):
    #     st.write(df_now_target2)
    #     st.caption('df_now_target2')

    #sales行の削除
    df_calc.drop(index='sales', inplace=True)
    df_calc = df_calc.fillna(0)
    # with st.expander('ターゲット得意先/シリーズ別売上/今期'):
    #     st.write(df_now_target2)
    #     st.caption('df_now_target2')

    st.markdown('####  ５．売れ筋展示品の抽出')
    #売れ筋の抽出
    #売れ筋の売り上げ下限を入力
    min_line = st.number_input('展示品の売上下限を入力', key='min_line', value=500000)

    #売上下限以下のdfを作成
    df_hot_sales = df_calc[df_calc[f'{target}_now'] >= min_line]

    df_hot_sales = df_hot_sales.sort_values(f'{target}_now', ascending=False)
    st.markdown('##### 売れ筋展示品')
    st.caption('df_hot_sales')
    st.write(df_hot_sales)
    
    #最近傍探索の準備　targetの最新のデータとdf_zenkoku3tをmerge
    df_zenkoku3t = df_zenkoku3.T #index:商品/col:得意先
    df_zenkoku3t.drop(target, axis=1, inplace=True)
    df_zenkoku3tm = df_calc.merge(df_zenkoku3t, left_index=True, right_index=True, how='outer')
    df_zenkoku4 = df_zenkoku3tm.T #index:得意先col:商品
    with st.expander('targetの最新のデータとdf_zenkoku3をmerge', expanded=False):
        st.write(df_zenkoku3tm)
        st.caption('df_zenkoku3tm')

    #**************************************************最近傍探索
    #インスタンス化
    #似ている得意先上位何位まで抽出するか
    st.markdown('#### ６．似ている商品の上位何位まで抽出するか')
    n_neighbors = st.number_input('抽出数', key='n_neighbors', value=7)

    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    #学習
    neigh.fit(df_zenkoku3tm)

    #売れ筋に近いアイテム上位何位まで抽出するか
    #ここで使うdfのindexが商品名になる。userベースとの違いはここだけ

    #売れ筋商品のリスト
    hot_item_list = list(df_hot_sales.index)
    #1個選定
    t_item = hot_item_list[0]
    st.write(hot_item_list)
    st.write(t_item)

    df_target = df_zenkoku3tm[df_zenkoku3tm.index==t_item]
    st.write(df_target)
    def knn(t_item):

        df_target = df_zenkoku3tm[df_zenkoku3tm.index==t_item]
        
        #指定した得意先に近い得意先を探索
        df_target = df_zenkoku3tm[df_zenkoku3tm.index==t_item]
        distance, indices = neigh.kneighbors(df_target ) #距離/indexナンバー

        #1次元リストに変換 np.flatten　元は2次元
        distance_f = distance.flatten()
        indices_f = indices.flatten()

        # with st.expander('distance_t/indices_t'):
        #     st.caption('distance_t')
        #     st.write(distance_f)
        #     st.caption('indices_t')
        #     st.write(indices_f)

        #indexオブジェクトからlist化
        index_list = list(df_zenkoku3tm.index)
        #indicesのint化
        indices_f = [int(x) for x in indices_f]

        #df化
        user_list = []
        for i in indices_f:
            index_name = index_list[i]
            user_list.append(index_name)

        df_result = pd.DataFrame(distance_f, columns=[t_item], index=user_list)
        
        return df_result

    #関数実行
    #一番売れている商品
    item1 = hot_item_list[0]
    df_item1 = knn(item1)
    st.write(df_item1)
    #二番目に売れている商品
    item2 = hot_item_list[1]
    df_item2 = knn(item2)
    st.write(df_item2)
    #3番目に売れている商品
    item3 = hot_item_list[2]
    df_item3 = knn(item3)
    st.write(df_item3)
    
    






            


