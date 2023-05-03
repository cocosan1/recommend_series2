import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import openpyxl
import plotly.figure_factory as ff
import plotly.graph_objects as go

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#*********************************************ユーザーベースの協調フィルタリング

#st
st.set_page_config(page_title='recommend_series')
st.markdown('### 展示品分析＆新規展示レコメンド/専門店')

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

st.divider()

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

    st.divider()    

    st.markdown('####  ３．展示品の選択')

    cate_list = list(df_target_sales.index)

    tenji_list = []
    # チェックボックスを表示し、ユーザーが選択したオプションを取得
    for cate in cate_list:
        if st.checkbox(cate):
            tenji_list.append(cate)

    st.divider()

    st.markdown('####  ４．受注ファイルの読み込み')

    # ***ファイルアップロード 今期***
    uploaded_file_now = st.file_uploader('今期受注ファイル', type='xlsx', key='now')

    if uploaded_file_now:
        df_now = pd.read_excel(
            uploaded_file_now, sheet_name='受注委託移動在庫生産照会', usecols=[3, 15, 16, 42, 50])  # index　ナンバー不要　index_col=0
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


    #sales行の削除
    df_calc.drop(index='sales', inplace=True)
    df_calc = df_calc.fillna(0)

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

    st.divider()
    #**************************************************最近傍探索
    #インスタンス化
    #似ている得意先上位何位まで抽出するか
  
    st.markdown('### ユーザーベース')
    st.markdown('#### 5．似ている得意先の上位何位まで抽出するか')
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

    #展示品のリスト diff順
    tenji_list2 = []
    for item in tenji_list:
        if item in list(df_merge_selected.index):
            tenji_list2.append(item)
        else:
            continue    

    df_tenji = df_merge_selected.loc[tenji_list2]
    df_tenji.sort_values('diff', inplace=True)
    with st.expander('展示品のリスト diff順'):
        st.write(df_tenji)
        st.caption('df_tenji')

    #非展示品のリスト 平均順
    nontenji_list = []
    for item in df_merge_selected.index:
        if item not in tenji_list:
            nontenji_list.append(item)


    df_nontenji = df_merge_selected.loc[nontenji_list]
    df_nontenji.sort_values('平均', ascending=False, inplace=True)

    #max minの追加
    df02 = df0.copy()
    df02['max'] = df02.max(axis=1)
    df02['min'] = df02.min(axis=1)

    df02 = df02[['max', 'min']]
    df_nontenjim = df_nontenji.merge(df02, left_index=True, right_index=True, how='left')

    with st.expander('非展示品のリスト diff順'):
        st.write(df_nontenjim)
        st.caption('df_nontenjim')

    st.divider()    

    #**********************************************************************アイテムベース
    #**************************************************最近傍探索
    #インスタンス化
    #似ている得意先上位何位まで抽出するか

    st.markdown('### アイテムベース')
    st.markdown('#### 似ているアイテムの上位何位まで抽出するか')
    n_neighbors = st.number_input('抽出数', key='n_neighbors2', value=10)

    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    #学習
    df_zenkoku4t = df_zenkoku4.T
    with st.expander('df_zenkoku4t', expanded=False):
        st.write(df_zenkoku4t)
    
    neigh.fit(df_zenkoku4t)

    #指定したアイテムに近いアイテムを探索
    #ここで使うdfのindexがアイテムになる。itemベースとの違いはここだけ

    
    def knn2(target_item):
        
        #指定した得意先に近い得意先を探索
        df_target = df_zenkoku4t[df_zenkoku4t.index==f'{target_item}']
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
        index_list = list(df_zenkoku4t.index)
        #indicesのint化
        indices_f = [int(x) for x in indices_f]

        #df化
        item_list = []
        for i in indices_f:
            index_name = index_list[i]
            item_list.append(index_name)

        df_result = pd.DataFrame(distance_f, columns=[f'{target_item}'], index=item_list)
        
        return df_result
    
    st.divider()
    #***************************************************************************結果

    st.markdown('## 結果')
    st.markdown('### 1.展示品の仕分け')

    #***************動いていない展示品
    #展示品の売り上げ上限を入力
    st.markdown('#### 1-1.動いていない展示品の抽出')
    max_line = st.number_input('いくら以下を抽出するか', key='max_line', value=100000)
    #展示品に絞込み
    df_cold_sales = df_calc.loc[tenji_list]
    #売上下限以下のdfを作成
    df_cold_sales = df_cold_sales[df_cold_sales[f'{target}_now'] <= max_line]
    df_cold_sales = df_cold_sales.sort_values(f'{target}_now', ascending=False)
    
    with st.expander('動いていない展示品', expanded=False):
        st.caption('df_cold_sales')
        st.write(df_cold_sales)

         #可視化
    #グラフを描くときの土台となるオブジェクト
    fig_cold = go.Figure()
    #今期のグラフの追加
    fig_cold.add_trace(
        go.Bar(
            x=df_cold_sales.index,
            y=df_cold_sales[f'{target}_now'],
            text=round(df_cold_sales[f'{target}_now']/10000),
            textposition="outside", 
            name='現在実績/今期')
    )

    #レイアウト設定     
    fig_cold.update_layout(
        title='動いていない展示品',
        showlegend=True #凡例表示
    )
    #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
    st.plotly_chart(fig_cold, use_container_width=True) 

    #****************売れ筋の売り上げ下限を入力
    st.markdown('#### 1-2.売れ筋の展示品')
    #売れているアイテム
    under_line = st.number_input('展示品の売上下限を入力', key='under_line', value=500000)

    #売上下限以下のdfを作成
    df_hot_sales = df_calc[df_calc[f'{target}_now'] >= under_line]

    df_hot_sales = df_hot_sales.sort_values(f'{target}_now', ascending=False)
    hot_item_list = list(df_hot_sales.index)
    #関数実行
    df_main = pd.DataFrame()
    for item in hot_item_list:
        df_knn = knn2(item)
        df_main = df_main.merge(df_knn, left_index=True, right_index=True, how='outer')

    with st.expander('売れ筋展示品との距離', expanded=False):
        st.markdown('###### 売れ筋展示品との距離/コサイン類似度')
        st.write(df_main)
        st.caption('df_main')

    #個数列の追加
    df_main['count'] = (df_main > 0).sum(axis=1)
    df_main.sort_values('count', ascending=False, inplace=True)
    #非展示品に絞る
    nontenji_list2 = []
    for item in df_main.index:
        if item not in tenji_list:
            nontenji_list2.append(item)
    df_main = df_main.loc[nontenji_list2]
    with st.expander('アイテムベース結果', expanded=False):
        st.write(df_main)
        st.caption('df_main')

    with st.expander('売れている展示品', expanded=False):
        st.caption('df_hot_sales')
        st.table(df_hot_sales)    

    #可視化
    #グラフを描くときの土台となるオブジェクト
    fig_hot = go.Figure()
    #今期のグラフの追加
    fig_hot.add_trace(
        go.Bar(
            x=df_hot_sales.index,
            y=df_hot_sales[f'{target}_now'],
            text=round(df_hot_sales[f'{target}_now']/10000),
            textposition="outside", 
            name='現在実績/今期')
    )

    #レイアウト設定     
    fig_hot.update_layout(
        title='売れている展示品',
        showlegend=True #凡例表示
    )
    #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
    st.plotly_chart(fig_hot, use_container_width=True)     

    st.divider()
    #**************************売れ筋の抽出
    
    st.markdown('### 2.現展示品の売上UPの可能性')
    st.markdown('#### 2-1.売上UPの可能性 〇')
    df_nobisiro = df_tenji[df_tenji['diff'] < -100000]    
    
    #***売上UPの可能性あり
    #受注データの日数の年間での割合算出
    diff_day = df_now['受注日'].max() - df_now['受注日'].min() #timedelta型
    diff_day = diff_day.days #timedeltaをintに変換
    #年換算比率
    rate_year = diff_day / 365

    #年間の数字に換算
    df_nobisiro[f'{target}_now'] = df_nobisiro[f'{target}_now'] /rate_year
    df_nobisiro['平均'] = df_nobisiro['平均'] /rate_year
    df_nobisiro['diff'] = df_nobisiro['diff'] /rate_year

    #int
    df_nobisiro[f'{target}_now'] = df_nobisiro[f'{target}_now'].astype('int')
    df_nobisiro['平均'] = df_nobisiro['平均'].astype('int')
    df_nobisiro['diff'] = df_nobisiro['diff'].astype('int')

    with st.expander('売上　現状予測/平均値', expanded=False):
        st.table(df_nobisiro)
        st.caption('df_nobisiro')
    
     #可視化
    #グラフを描くときの土台となるオブジェクト
    fig_nobishiro = go.Figure()
    #今期のグラフの追加
    fig_nobishiro.add_trace(
        go.Bar(
            x=df_nobisiro.index,
            y=df_nobisiro[f'{target}_now'],
            text=round(df_nobisiro[f'{target}_now']/10000),
            textposition="outside", 
            name='現状予測')
    )
    #前期のグラフの追加
    fig_nobishiro.add_trace(
        go.Bar(
            x=df_nobisiro.index,
            y=df_nobisiro['平均'],
            text=round(df_nobisiro['平均']/10000),
            textposition="outside", 
            name='平均値/可能性'
            )
    )
    #レイアウト設定     
    fig_nobishiro.update_layout(
        title='売上　現状予測/平均値/年間',
        showlegend=True #凡例表示
    )
    #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
    st.plotly_chart(fig_nobishiro, use_container_width=True) 
    st.caption('● 数字は今期受注データの期間から年間に換算した予測数字')
    st.caption('● ユーザーベースから')


    #***売上UPの可能性低
    st.markdown('#### 2-2.売上UPの可能性 ▲')
    df_nonnobisiro = df_tenji[df_tenji['diff'] > 0]

    #年間の数字に換算
    df_nonnobisiro[f'{target}_now'] = df_nonnobisiro[f'{target}_now'] /rate_year
    df_nonnobisiro['平均'] = df_nonnobisiro['平均'] /rate_year
    df_nonnobisiro['diff'] = df_nonnobisiro['diff'] /rate_year

    #int
    df_nonnobisiro[f'{target}_now'] = df_nonnobisiro[f'{target}_now'].astype('int')
    df_nonnobisiro['平均'] = df_nonnobisiro['平均'].astype('int')
    df_nonnobisiro['diff'] = df_nonnobisiro['diff'].astype('int')

    with st.expander('売上　現状予測/平均値', expanded=False):
        st.table(df_nonnobisiro)
        st.caption('df_nonnobisiro')

     #可視化
    #グラフを描くときの土台となるオブジェクト
    fig_nonnobishiro = go.Figure()
    #今期のグラフの追加
    fig_nonnobishiro.add_trace(
        go.Bar(
            x=df_nonnobisiro.index,
            y=df_nonnobisiro[f'{target}_now'],
            text=round(df_nonnobisiro[f'{target}_now']/10000),
            textposition="outside", 
            name='現状予測')
    )
    #前期のグラフの追加
    fig_nonnobishiro.add_trace(
        go.Bar(
            x=df_nonnobisiro.index,
            y=df_nonnobisiro['平均'],
            text=round(df_nonnobisiro['平均']/10000),
            textposition="outside", 
            name='平均値/可能性'
            )
    )
    #レイアウト設定     
    fig_nonnobishiro.update_layout(
        title='売上　現状予測/平均値/年間',
        showlegend=True #凡例表示
    )
    #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
    st.plotly_chart(fig_nonnobishiro, use_container_width=True) 
    st.caption('● 数字は今期受注データの期間から年間に換算した予測数字')
    st.caption('● ユーザーベースから')

    st.divider()

    #**********************************************レコメンド
    #新展示レコメンド
    st.markdown('### 3.新規展示レコメンド')

    #ユーザーベース
    st.markdown('#### 3-1.ユーザーベース')
    #年間の数字に換算
    df_nontenjim[f'{target}_now'] = df_nontenjim[f'{target}_now'] /rate_year
    df_nontenjim['平均'] = df_nontenjim['平均'] /rate_year
    df_nontenjim['diff'] = df_nontenjim['diff'] /rate_year

    #int
    df_nontenjim[f'{target}_now'] = df_nontenjim[f'{target}_now'].astype('int')
    df_nontenjim['平均'] = df_nontenjim['平均'].astype('int')
    df_nontenjim['diff'] = df_nontenjim['diff'].astype('int')

    with st.expander('展示品レコメンド', expanded=False):
        st.table(df_nontenjim)
        st.caption('● 数字は今期受注データの期間から年間に換算した予測数字')
        st.caption('● max/minは年間の類似得意先の売上から抽出')

      #可視化
    #グラフを描くときの土台となるオブジェクト
    fig_user = go.Figure()
    #今期のグラフの追加
    fig_user.add_trace(
        go.Bar(
            x=df_nontenjim.index,
            y=df_nontenjim['平均'].head(),
            text=round(df_nontenjim['平均'].head()/10000),
            textposition="outside", 
            name='平均値')
    )
    #前期のグラフの追加
    fig_user.add_trace(
        go.Bar(
            x=df_nontenjim.index,
            y=df_nontenjim['max'].head(),
            text=round(df_nontenjim['max'].head()/10000),
            textposition="outside", 
            name='max/可能性'
            )
    )
    #レイアウト設定     
    fig_user.update_layout(
        title='新規展示品レコメンド/予測年間',
        showlegend=True #凡例表示
    )
    #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
    st.plotly_chart(fig_user, use_container_width=True) 

    st.divider()

    #****************************************アイテムベース
    st.markdown('#### アイテムベースベース')

    with st.expander('レコメンド展示品', expanded=False):
        st.table(df_main)
        st.caption('df_main')

    #可視化
    #グラフを描くときの土台となるオブジェクト
    fig_item = go.Figure()
    #今期のグラフの追加
    for col in df_main.columns[:-1]:
        fig_item.add_trace(
            go.Bar(
                x=df_main.index,
                y=df_main[col].head(),
                text=round(df_main[col].head(), 2),
                textposition="outside", 
                name=col)
    )

    #レイアウト設定     
    fig_item.update_layout(
        title='新規展示品レコメンド/売れ筋展示品との近さ',
        showlegend=True #凡例表示
    )
    #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
    st.plotly_chart(fig_item, use_container_width=True) 
    st.caption('● 左から推奨順')
    st.caption('● グラフの本数が多いほど推奨度が高い')
    st.caption('● 数値はベクトルの距離/小さいほど近い')
    



      

