import pandas as pd
import datetime
import streamlit as st


#************************************************************
#index商品分類/col得意先/val売上
@st.cache_data(ttl=datetime.timedelta(hours=1))
def make_data_cust(df):
    #LD分類カラムの新設
    cate_list = []
    for cate in df['商品分類名2']:
        if cate in ['ダイニングテーブル', 'ダイニングチェア', 'ベンチ']:
            cate_list.append('d')
        elif cate in ['リビングチェア', 'クッション', 'リビングテーブル']:
            cate_list.append('l')
        else:
            cate_list.append('none') 

    df['category'] = cate_list
    #シリーズ名＋LD　カラムの新設
    df['series2'] = df['category'] + '_' + df['シリーズ名']

    #伝票NOカラムの作成
    df['伝票番号2'] = df['伝票番号'].map(lambda x: x[:8])

    #必要なカラムだけに絞る
    df = df[['得意先名', '金額', 'series2', '伝票番号2']]

    # #series2のリスト作成
    # a_list = []
    # for a in df['series2'].unique():
    #     if 'none' in a:
    #         continue
    #     elif 'その他' in a:
    #         continue
    #     else:
    #         a_list.append(a)
    #     a_list.sort() 
    
    #商品リストの絞込み　手作業
    selected_list = ['l_森のことば',
    'l_穂高',
    'd_侭 JIN',
    'd_SEOTO',
    'd_L-CHAIR',
    'd_クレセント',
    'd_森のことば',
    'd_TUGUMI',
    'd_森のことばIBUKI',
    'l_CHIGUSA(ﾁｸﾞｻ）',
    'd_風のうた',
    'l_VIOLA (ｳﾞｨｵﾗ)',
    'l_SEOTO',
    'l_風のうた',
    'd_ALMO (ｱﾙﾓ)',
    'd_nae',
    'd_穂高',
    'd_PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）',
    'd_VIOLA (ｳﾞｨｵﾗ)',
    'l_ｽﾀﾝﾀﾞｰﾄﾞｺﾚｸｼｮﾝ',
    'd_YURURI',
    'l_SEION 静穏',
    'd_SEOTO-EX',
    'd_BAGUETTE LB(ﾊﾞｹｯﾄｴﾙﾋﾞｰ)',
    'd_tsubura',
    'l_杜の詩',
    'd_ﾆｭｰﾏｯｷﾝﾚｲ',
    'd_COBRINA',
    'l_YURURI',
    'd_Northern Forest',
    'l_nae',
    'l_PRESCELTO (ﾌﾟﾚｼｪﾙﾄ）',
    'l_森のことばIBUKI',
    'd_AWASE',
    'l_AWASE',
    'd_CHIGUSA(ﾁｸﾞｻ）',
    'd_HIDA',
    'd_Kinoe',
    'l_Northern Forest',
    'l_Kinoe',
    'd_円空',
    'l_ﾆｭｰﾏｯｷﾝﾚｲ',
    'l_TUGUMI',
    'l_森のことば ウォルナット',
    'l_SEOTO-EX',
    'd_杜の詩',
    'l_ALMO (ｱﾙﾓ)',
    'd_侭SUGI',
    'd_森のことば ウォルナット',
    'l_COBRINA',]
    selected_list.sort()

    #得意先毎に集計
    df_calc = pd.DataFrame(index=selected_list)
    for cust in df['得意先名'].unique():
        val_list = []
        df_cust = df[df['得意先名']==cust]
        for cate in selected_list:
            df_cust2 = df_cust[df_cust['series2']==cate]
            if len(df_cust2) == 0:
                val_list.append(0)
            else:
                val = df_cust2['金額'].sum()
                val_list.append(val)
        df_temp = pd.DataFrame(val_list, columns=[cust], index=selected_list)
        df_calc = df_calc.merge(df_temp, left_index=True, right_index=True, how='outer') 

    df_calc.loc['sales'] = df_calc.sum()
    
    return df_calc
#*******************************************************************************item
@st.cache_data(ttl=datetime.timedelta(hours=1))
def make_data_item(df):
    #dfをtargetで絞込み

    #LD分類
    cate_list = []
    for cate in df['商品分類名2']:
        if cate in ['ダイニングテーブル', 'ダイニングチェア', 'ベンチ']:
            cate_list.append('d')
        elif cate in ['リビングチェア', 'クッション', 'リビングテーブル']:
            cate_list.append('l')
        else:
            cate_list.append('none') 
    
    #分類列を追加
    df['category'] = cate_list 
    #noneを削除
    df = df[df['category']!='none']
    #シリーズ名＋LD分類
    df['series2'] = df['category'] + '_' + df['シリーズ名']

    #必要なカラムに絞る
    df = df[['得意先名', '金額', 'series2']] 

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
        df_cate2 = df[df['series2']==cate]
        if len(df_cate2) == 0:
            val_list.append(0)
        else:
            val = df_cate2['金額'].sum()
            val_list.append(val)
    df_calc = pd.DataFrame(val_list, columns=['合計'], index=selected_list)

    df_calc.loc['sales'] = df_calc.sum()

    return df_calc
#**********************************************************************************item target
@st.cache_data(ttl=datetime.timedelta(hours=1))
def make_data_item_target(target, df):
    #dfをtargetで絞込み
    df = df[df['得意先名'] == target]

    # target_name = f'{target}_new'
    # df.rename(columns={'得意先名', target_name})
    #LD分類
    cate_list = []
    for cate in df['商品分類名2']:
        if cate in ['ダイニングテーブル', 'ダイニングチェア', 'ベンチ']:
            cate_list.append('d')
        elif cate in ['リビングチェア', 'クッション', 'リビングテーブル']:
            cate_list.append('l')
        else:
            cate_list.append('none') 
    
    #分類列を追加
    df['category'] = cate_list 

    #noneを削除
    df = df[df['category']!='none']

    #シリーズ名＋LD分類
    df['series2'] = df['category'] + '_' + df['シリーズ名']

    #必要なカラムに絞る
    df = df[['得意先名', '金額', 'series2']] 

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
    

    val_list = []
    for cate in selected_list:
        df_cust2 = df[df['series2']==cate]
        if len(df_cust2) == 0:
            val_list.append(0)
        else:
            val = df_cust2['金額'].sum()
            val_list.append(val)
    df_calc = pd.DataFrame(val_list, columns=[f'{target}_now'], index=selected_list)

    df_calc.loc['sales'] = df_calc.sum()

    return df_calc
