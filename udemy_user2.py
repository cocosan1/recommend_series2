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
    st.table(df_hot_sales)
    
    #最近傍探索の準備　targetの最新のデータとdf_zenkoku3tをmerge
    df_zenkoku3t = df_zenkoku3.T #index:商品/col:得意先
    df_zenkoku3t.drop(target, axis=1, inplace=True)
    df_zenkoku3tm = df_calc.merge(df_zenkoku3t, left_index=True, right_index=True, how='outer')
    df_zenkoku4 = df_zenkoku3tm.T #index:得意先col:商品
    with st.expander('targetの最新のデータとdf_zenkoku3をmerge', expanded=False):
        st.write(df_zenkoku4)
        st.caption('df_zenkoku4')

    with st.expander('df_zenkoku4のdescribe'):
        st.write(df_zenkoku4.describe())
        st.caption('df_zenkoku4のdescribe')
        st.write((df_zenkoku4 > 0).sum())
        st.caption('df_zenkoku4の0超えの値の数')

    #**************************************************最近傍探索
    #インスタンス化
    #似ている得意先上位何位まで抽出するか
    st.markdown('#### ６．似ている得意先の上位何位まで抽出するか')
    n_neighbors = st.number_input('抽出数', key='n_neighbors', value=5)

    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    #学習
    neigh.fit(df_zenkoku4)

    #指定した得意先に近い得意先を探索
    #ここで使うdfのindexが得意先になる。itemベースとの違いはここだけ
    
    def knn():
        
        #指定した得意先に近い得意先を探索
        df_target = df_zenkoku4[df_zenkoku4.index==f'{target}_now']
        distance, indices = neigh.kneighbors(df_target ) #距離/indexナンバー

        #1次元リストに変換 np.flatten　元は2次元
        distance_f = distance.flatten()
        indices_f = indices.flatten()

        with st.expander('distance_t/indices_t'):
            st.caption('distance_t')
            st.write(distance_f)
            st.caption('indices_t')
            st.write(indices_f)

        #indexオブジェクトからlist化
        index_list = list(df_zenkoku4.index)
        #indicesのint化
        indices_f = [int(x) for x in indices_f]

        #df化
        user_list = []
        for i in indices_f:
            index_name = index_list[i]
            user_list.append(index_name)

        df_result = pd.DataFrame(distance_f, columns=[f'{target}_now'], index=user_list)
        
        return df_result

    #関数実行
    df_knn = knn()

    with st.expander('売れ筋展示品との距離', expanded=False):
        st.markdown('###### 売れ筋展示品との距離/コサイン類似度')
        st.write(df_knn)
        st.caption('df_knn')
    
    def comparison_cust(index_no):
        #1番似ている得意先名抽出
        cust1 = df_knn.index[index_no]
        
        #1番似ている得意先のitem別売上
        s1 = df_zenkoku4.loc[cust1]
        #target得意先のitem別売上
        s_target = df_zenkoku4.loc[f'{target}_now']
        #mergeするためdf化
        df1 = pd.DataFrame(s1)
        df_target = pd.DataFrame(s_target)
        #merge
        df_merge = df_target.merge(df1, left_index=True, right_index=True, how='outer')
        # df_merge['rate'] = df_merge[f'{target}_now'] / df_merge[cust1]

        return df_merge
    
    #似ている得意先トップ３の平均df作る
    df0 = pd.DataFrame()
    for i in range(1, 4, 1):
        df = comparison_cust(i)
        #似ている得意先名の抽出
        cust = df.columns[1]
        df_cust= df[cust]
        df0 =df0.merge(df_cust, left_index=True, right_index=True, how='outer')
    with st.expander('一覧 low_data', expanded=False):
        st.table(df0) 
        st.caption('df0')

    #似ている得意先毎にtargetとスケールを合わせたdfを作る
    df_rate = pd.DataFrame()
    for col in df0.columns:
        rate = df_calc[f'{target}_now'].sum() / df0[col].sum()
        df_temp = df0[col] * rate
        df_temp = df_temp.astype('int')
        
        df_rate = df_rate.merge(df_temp, left_index=True, right_index=True, how='outer')

    df_rate['平均'] = df_rate.mean(axis=1)
    df_rate['平均'] = df_rate['平均'].astype('int')   

    with st.expander('targetとスケールを合わせた'):
     st.table(df_rate)
     st.caption('df_rate') 

    #targetと平均のmerge
    df_mean = pd.DataFrame(df_rate['平均'])
    df_merge = df_calc.merge(df_mean, left_index=True, right_index=True, how='outer')
    with st.expander('targetと平均のmerge'):
        st.write(df_merge)
        st.caption('df_merge')

    #10万以下のitemの削除
    #削除ライン　targetの売上合計の2％
    cut_line = df_calc[f'{target}_now'].sum() *0.02
    df_merge_selected = df_merge[(df_merge[f'{target}_now'] > cut_line) | (df_merge['平均'] > cut_line)]
    #round floatのケタ数指示可
    df_merge_selected['rate'] = round(df_merge_selected[f'{target}_now'] / df_merge_selected['平均'], 2)
    df_merge_selected['diff'] = df_merge_selected[f'{target}_now'] - df_merge_selected['平均']
    with st.expander('削除ライン　targetの売上合計の2％'):
        st.write(df_merge_selected)
        st.caption('df_merge_selected')

    st.write('展示品のリスト diff順')
    tenji_list2 = []
    for item in tenji_list:
        if item in list(df_merge_selected.index):
            tenji_list2.append(item)
        else:
            continue    

    df_tenji = df_merge_selected.loc[tenji_list2]
    df_tenji.sort_values('diff', inplace=True)
    st.write(df_tenji)

    st.write('非展示品のリスト diff順')
    nontenji_list = []
    for item in df_merge_selected.index:
        if item not in tenji_list:
            nontenji_list.append(item)


    df_nontenji = df_merge_selected.loc[nontenji_list]
    df_nontenji.sort_values('diff', inplace=True)

    #max minの追加
    df02 = df0.copy()
    df02['max'] = df02.max(axis=1)
    df02['min'] = df02.min(axis=1)

    df02 = df02[['max', 'min']]
    df_nontenjim = df_nontenji.merge(df02, left_index=True, right_index=True, how='left')

    st.write(df_nontenjim)



   
        



    # st.markdown('###### 一番似ている得意先との比較')
    # df1 = comparison_cust(1)
    # df1.loc['合計'] = df1.sum(axis=0)
    # st.write(df1)

    # #売れている商品トップ１０に絞る　似ている店ベース
    # cust = df1.columns[1]
    # df1.sort_values(cust, ascending=False, inplace=True)
    # df1_10 = df1[:11]
    # st.write(df1_10)

    # st.markdown('###### 2番似ている得意先との比較')
    # df2 = comparison_cust(2)
    # df2.loc['合計'] = df2.sum(axis=0)

    # #売れている商品トップ１０に絞る　似ている店ベース
    # cust2 = df2.columns[1]
    # df2.sort_values(cust2, ascending=False, inplace=True)
    # df2_10 = df2[:11]
    # st.write(df2_10)






    # #売上対比
    # sales_rate = df1.iloc[-1, 0] / df1.iloc[-1, 1]
    # st.write('売上比較　target/一番似ている得意先')
    # st.write(sales_rate)

    # #df1の商品絞込み
    # cust1 = df_knn.index[1]
    # #両方が10万以下をカット
    # df2 = df1[(df1[f'{target}_now'] > 100000) | (df1[cust1] > 100000)]
    # df2.sort_values(cust1, ascending=False, inplace=True)

    # df2['調整売上/一番似ている得意先'] = round(df2[cust1] * sales_rate)
    # df2['差額/調整後'] = df2['調整売上/一番似ている得意先'] - df2[f'{target}_now']
    # df2.sort_values('差額/調整後', ascending=False, inplace=True)
    # st.dataframe(df2)
    # st.caption('両方が10万以下の商品をカット')

    # st.write('展示品の伸び代を見る')
    # df2_tenji = df2.loc[tenji_list]
    # df2_tenji.sort_values('差額/調整後', ascending=False, inplace=True)
    # st.dataframe(df2_tenji)

    # st.write('非展示品の予測売上を見る')

    # #非展示リストの作成
    # non_tenji_list = list(df2.index) 

    # for item in tenji_list:
    #     non_tenji_list.remove(item) #展示アイテムを削除

    # df2_nontenji = df2.loc[non_tenji_list]
    # df2_nontenji.sort_values('差額/調整後', ascending=False, inplace=True)
    # st.dataframe(df2_nontenji)


    # st.markdown('###### 基準rateの設定')
    # st_rate = st.number_input('基準rate', key='st_rate')

    # st.markdown('###### 上限rateの設定')
    # up_rate = st.number_input('上限rate', key='up_rate')


    # st.markdown('###### 上限rate以下のアイテム表示')
    # df1_low = df1[df1['rate'] <= up_rate]
    

    # cust1 = df_knn.index[1]
    # df1_low['基準rate売上'] = round(df1_low['金額']*(st_rate/ df1_low['rate']))
    # df1_low['差額'] = df1_low['基準rate売上'] - df1_low['金額']
    # st.dataframe(df1_low)





            


