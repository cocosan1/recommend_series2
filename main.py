import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import openpyxl

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from pandas.core.common import random_state
# import lightgbm as lgb
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import mean_squared_error # モデル評価用(平均二乗誤差)
# from sklearn.metrics import r2_score # モデル評価用(決定係数)

#st
st.set_page_config(page_title='recommend_series')
st.markdown('### レコメンド アプリ/専門店')

#データ読み込み
df_zenkoku = pd.read_pickle('df_zenkoku7879.pickle')

st.markdown('###### １．分析対象得意先の絞込み')
#得意先範囲の設定
sales_max = st.number_input('分析対象得意先の売上上限を入力', key='sales_max', value=70000000)
sales_min = st.number_input('分析対象得意先の売上下限を入力', key='sales_min', value=2000000)

#salesによる絞込み
df_zenkoku2 = df_zenkoku.copy()
df_zenkoku2 = df_zenkoku2[(df_zenkoku2['sales'] >= sales_min) & (df_zenkoku2['sales'] <= sales_max)]
st.caption(f'対象得意先数: {len(df_zenkoku2)}')

img_yajirusi = Image.open('矢印.jpeg')
st.image(img_yajirusi, width=20)

#target選定用リスト
st.markdown('######  ２．target得意先の選択')
cust_text = st.text_input('得意先名の一部を入力 例）ケンポ')

target_list = []
for cust_name in df_zenkoku2.index:
    if cust_text in cust_name:
        target_list.append(cust_name)

if target_list != '':
    # selectbox target ***
    target = st.selectbox(
        'target得意先:',
        target_list,   
    ) 
st.image(img_yajirusi, width=20)  

if target != '':

    #dfをアイテム列だけに絞る
    df_zenkoku3 = df_zenkoku2.drop(['sales', 'a_price'], axis=1)

    #相関係数 相関なし0-0.2 やや相関あり0.2-0.4 相関あり 0.4-0.7 強い相関 0.7-
    df_corr = df_zenkoku3.corr()
    with st.expander('相関係数表', expanded=False):
        st.write(df_corr)

    #targetの売上を整理
    df_zenkoku_temp = df_zenkoku.drop(['sales', 'a_price'], axis=1)
    df_target_sales = df_zenkoku_temp.loc[target] #indexがシリーズ名/カラム名が店名のseries

    with st.expander('シリーズ別売上/年間', expanded=False):
        st.write(df_target_sales)

    #******************アイテムベース*******************************
    #recomenndリスト作成のための計算
    #相関係数×シリーズ売上の一覧
    sim_candidates = pd.Series()
    for i in range(0, len(df_target_sales.index)):
        # print('adding sims for ' + target_sales.index[i] + '...')
        #シリーズ毎に相関表作成 series型
        sims = df_corr[df_target_sales.index[i]]
        #相関係数×シリーズ金額
        sims = sims.map(lambda x: round(x * df_target_sales[i]))
        sim_candidates = sim_candidates.append(sims)

    #シリーズ毎に合計
    sim_candidates = sim_candidates.groupby(sim_candidates.index).sum()
    sim_candidates.sort_values(ascending=False, inplace=True)

    with st.expander('相関係数×売上→シリーズ毎に集計', expanded=False):
        st.write(sim_candidates)

    #pointsとsalesをmerge
    df_simcan = pd.DataFrame(sim_candidates)
    df_sales = pd.DataFrame(df_target_sales)
    df_merge = df_simcan.merge(df_sales, left_index=True, right_index=True, how='right')
    df_merge = df_merge.sort_values(0, ascending=False)
    df_merge.columns = ['points', 'sales'] 

    with st.expander('pointとsalesを一覧化', expanded=False):
        st.write(df_merge)  

    st.markdown('######  ３．展示品の選択')

    #展示品の指定
    tenji_series = st.multiselect(
        '展示品を選択',
        df_target_sales.index)

    st.image(img_yajirusi, width=20) 

    st.markdown('######  ４．動きのよくない展示品の抽出')

    # ***ファイルアップロード 今期***
    uploaded_file_now = st.file_uploader('今期', type='xlsx', key='now')

    if uploaded_file_now:
        df_now = pd.read_excel(
            uploaded_file_now, sheet_name='受注委託移動在庫生産照会', usecols=[3, 15, 42, 50])  # index　ナンバー不要　index_col=0
    else:
        st.info('今期のファイルを選択してください。')
        st.stop()

    #targetから79などを削除
    target2 = target[:-2]
    
    #LD分類
    df_now_target = df_now[df_now['得意先名']==target2]
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
    df_now_target['series2'] = df_now_target['シリーズ名'] + '_' + df_now_target['category']

    with st.expander('target得意先に絞った表', expanded=False):
        st.write(df_now_target) 

    #seiries2シリーズ名＋LD分類を品番に変換

    index_list = []
    htsd_list = []
    kxd_list = []
    sgd_list = []
    kdd_list = []
    snd_list = []
    vzd_list = []
    sld_list = []
    fxd_list = []
    rkd_list = []
    psd_list = []

    snl_list = []
    hkl_list = []
    wkl_list = []
    kdl_list = []
    wql_list = []
    wnl_list = []
    fxl_list = []
    psl_list = []
    sdl_list = []

    sales_list = []

    for cust in df_now_target['得意先名'].unique():
        index_list.append(cust)
        df = df_now_target[df_now_target['得意先名']==cust]
        sales = df['金額'].sum()
        sales_list.append(sales)

        for series in  ['侭 JIN_d', 'SEOTO-EX_d', 'クレセント_d', 'SEOTO_d', '森のことば_d', 'TUGUMI_d', 'YURURI_d',\
            '風のうた_d', 'ALMO (ｱﾙﾓ)_d', 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_d',\
            '森のことば_l', '穂高_l', 'CHIGUSA(ﾁｸﾞｻ）_l', 'SEOTO_l', 'SEION 静穏_l', 'VIOLA (ｳﾞｨｵﾗ)_l',\
            '風のうた_l', 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_l', 'ｽﾀﾝﾀﾞｰﾄﾞｺﾚｸｼｮﾝ_l']:

                    if len(df_now_target[df_now_target['series2']==series]) == 0:
                        if series == '侭 JIN_d':
                            htsd_list.append(0)
                        elif series == 'SEOTO-EX_d':  
                            kxd_list.append(0)
                        elif series == 'クレセント_d':
                            sgd_list.append(0)
                        elif series == 'SEOTO_d':
                            kdd_list.append(0)
                        elif series == '森のことば_d':
                            snd_list.append(0)
                        elif series == 'TUGUMI_d':
                            vzd_list.append(0) 
                        elif series == 'YURURI_d':
                            sld_list.append(0)
                        elif series == '風のうた_d':
                            fxd_list.append(0)
                        elif series == 'ALMO (ｱﾙﾓ)_d':
                            rkd_list.append(0)
                        elif series == 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_d':
                            psd_list.append(0)        

                        elif series == '森のことば_l':
                            snl_list.append(0)
                        elif series == '穂高_l':
                            hkl_list.append(0)
                        elif series == 'CHIGUSA(ﾁｸﾞｻ）_l':
                            wkl_list.append(0)       
                        elif series == 'SEOTO_l':
                            kdl_list.append(0)
                        elif series == 'SEION 静穏_l':
                            wql_list.append(0) 
                        elif series == 'VIOLA (ｳﾞｨｵﾗ)_l':
                            wnl_list.append(0) 
                        elif series == '風のうた_l':
                            fxl_list.append(0) 
                        elif series == 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_l':
                            psl_list.append(0)
                        elif series == 'ｽﾀﾝﾀﾞｰﾄﾞｺﾚｸｼｮﾝ_l':
                            sdl_list.append(0)          
                            
                    else:
                        sales = df_now_target[df_now_target['series2']==series]['金額'].sum()
                        if series == '侭 JIN_d':
                            htsd_list.append(sales)
                        elif series == 'SEOTO-EX_d':  
                            kxd_list.append(sales)
                        elif series == 'クレセント_d':
                            sgd_list.append(sales)
                        elif series == 'SEOTO_d':
                            kdd_list.append(sales)
                        elif series == '森のことば_d':
                            snd_list.append(sales)
                        elif series == 'TUGUMI_d':
                            vzd_list.append(sales) 
                        elif series == 'YURURI_d':
                            sld_list.append(sales)
                        elif series == '風のうた_d':
                            fxd_list.append(sales)
                        elif series == 'ALMO (ｱﾙﾓ)_d':
                            rkd_list.append(sales)
                        elif series == 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_d':
                            psd_list.append(sales)        
                        elif series == '森のことば_l':
                            snl_list.append(sales)
                        elif series == '穂高_l':
                            hkl_list.append(sales)
                        elif series == 'CHIGUSA(ﾁｸﾞｻ）_l':
                            wkl_list.append(sales)       
                        elif series == 'SEOTO_l':
                            kdl_list.append(sales)
                        elif series == 'SEION 静穏_l':
                            wql_list.append(sales) 
                        elif series == 'VIOLA (ｳﾞｨｵﾗ)_l':
                            wnl_list.append(sales) 
                        elif series == '風のうた_l':
                            fxl_list.append(sales) 
                        elif series == 'PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）_l':
                            psl_list.append(sales)
                        elif series == 'ｽﾀﾝﾀﾞｰﾄﾞｺﾚｸｼｮﾝ_l':
                            sdl_list.append(sales)

    df_now_target2 =pd.DataFrame(list(zip(sales_list, htsd_list, kxd_list, sgd_list, kdd_list, snd_list,\
                                  vzd_list, sld_list, fxd_list, rkd_list, psd_list,\
                                  snl_list, hkl_list, wkl_list, kdl_list, wql_list, wnl_list, fxl_list,\
                                  psl_list, sdl_list)), \
                         index=['売上'],\
                          columns=['sales', 'hts_d', 'kx_d', 'sg_d', 'kd_d', 'sn_d', 'vz_d', 'sl_d',\
                                   'fx_d', 'rk_d', 'ps_d',\
                                   'sn_l', 'hk_l', 'wk_l', 'kd_l', 'wq_l', 'wn_l', 'fx_l', 'ps_l', 'sd_l']).T
    
    with st.expander('得意先別/シリーズLD別/売上表', expanded=False):
        st.write(df_now_target2)

    #展示品に絞る
    df_now_tenji = df_now_target2.loc[tenji_series]

    #展示品の売り上げ下限を入力
    min_line = st.number_input('展示品の売上下限を入力', key='min_line', value=0)

    #売上下限以下のdfを作成
    df_problem_series = df_now_tenji[df_now_tenji['売上'] <= min_line]

    df_problem_series = df_problem_series.sort_values('売上')

    st.write('動きの良くない展示シリーズ')
    st.table(df_problem_series) 

    #******************ユーザーベース*******************************
    #データの正規化
    df_zenkoku3_norm = df_zenkoku3.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)),axis=1)
    #axis=1 行方向
    df_zenkoku3_norm = df_zenkoku3_norm.fillna(0)   
      