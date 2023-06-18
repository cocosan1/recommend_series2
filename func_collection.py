import pandas as pd
import datetime
import streamlit as st
import os
import plotly.graph_objects as go
# import func_collection as fc


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

#########################################################################前処理　df_now2/last2
def pre_processing(df_now, df_last, selected_base, selected_cate):
    if selected_cate == 'リビングチェア':
        # 含まない文字列のリストを作成
        df_lnow = df_now[df_now['商品分類名2']=='リビングチェア']
        df_llast = df_last[df_last['商品分類名2']=='リビングチェア']

        exclude_strs = ['ｽﾂｰﾙ', 'ｵｯﾄﾏﾝ', 'ﾛｯｷﾝｸﾞﾁｪｱ', '肘木', 'ﾊﾟｰｿﾅﾙﾁｪｱ']
        exclude_pattern = '|'.join(exclude_strs)

        df_now2 = df_lnow[~df_lnow['商　品　名'].str.contains(exclude_pattern, regex=True)]
        df_last2 = df_llast[~df_llast['商　品　名'].str.contains(exclude_pattern, regex=True)]

        df_now2['品番'] = df_now2['商　品　名'].apply(lambda x: x.split(' ')[0])
        df_last2['品番'] = df_last2['商　品　名'].apply(lambda x: x.split(' ')[0])

        df_now2[selected_base] = df_now2[selected_base].fillna(0)
        df_last2[selected_base] = df_last2[selected_base].fillna(0)

        df_now2['受注年月'] = df_now2['受注日'].dt.strftime("%Y-%m")
        df_now2['受注年月'] = pd.to_datetime(df_now2['受注年月'])

        df_last2['受注年月'] = df_last2['受注日'].dt.strftime("%Y-%m")
        df_last2['受注年月'] = pd.to_datetime(df_last2['受注年月'])

    else:
        df_now2 = df_now[df_now['商品分類名2']==selected_cate]
        df_last2 = df_last[df_last['商品分類名2']==selected_cate]

        df_now2['品番'] = df_now2['商　品　名'].apply(lambda x: x.split(' ')[0])
        df_last2['品番'] = df_last2['商　品　名'].apply(lambda x: x.split(' ')[0])

        df_now2[selected_base] = df_now2[selected_base].fillna(0)
        df_last2[selected_base] = df_last2[selected_base].fillna(0)

        df_now2['受注年月'] = df_now2['受注日'].dt.strftime("%Y-%m")
        df_now2['受注年月'] = pd.to_datetime(df_now2['受注年月'])

        df_last2['受注年月'] = df_last2['受注日'].dt.strftime("%Y-%m")
        df_last2['受注年月'] = pd.to_datetime(df_last2['受注年月'])
    
    return df_now2, df_last2
###############################################################################分布df作成
def make_bunpu(df_now2, selected_base):
    #数量データ分布一覧/itemベース
    items = []
    cnts = []
    quan25s = []
    medis = []
    quan75s = []
    quan90s = []
    maxs = []
    span2575s = []
    den_cnts = []
    den_rates = []
    for item in df_now2['品番'].unique():
        items.append(item)
        df_item = df_now2[df_now2['品番']==item]
        s_cust = df_item.groupby('得意先名')[selected_base].sum()
        
        cnt = s_cust.count()
        quan25 = round(s_cust.quantile(0.25), 1)
        medi = s_cust.median()
        quan75 = round(s_cust.quantile(0.75), 1)
        quan90 = round(s_cust.quantile(0.9), 1)
        max_num = s_cust.max()
        span2575 = quan75 - quan25
        den_cnt = df_item['伝票番号2'].nunique()
        den_rate = round(den_cnt / cnt, 1)

        cnts.append(cnt)
        quan25s.append(quan25)
        medis.append(medi)
        quan75s.append(quan75)
        quan90s.append(quan90)
        maxs.append(max_num)
        span2575s.append(span2575)
        den_cnts.append(den_cnt)
        den_rates.append(den_rate)

    df_calc = pd.DataFrame(list(zip(cnts, quan25s, medis, quan75s, quan90s, maxs, span2575s, den_cnts, \
                                    den_rates)), \
                    columns=['得意先数', '第2四分位', '中央値', '第3四分位', '上位10%', '最大値', 'span2575', \
                             '伝票数', '伝票数/得意先数'], index=items)

    return df_calc

#*******************************************************************グラフ
class Graph():
        def make_bar(self, val_list, x_list):
            #可視化
            #グラフを描くときの土台となるオブジェクト
            fig = go.Figure()
            #今期のグラフの追加
            for (val, x) in zip(val_list, x_list):
                fig.add_trace(
                    go.Bar(
                        x=[x],
                        y=[val],
                        text=[round(val/10000) if int(val) >= 10000 else int(val)],
                        textposition="outside", 
                        name=x)
                )
            #レイアウト設定     
            fig.update_layout(
                showlegend=False #凡例表示
            )
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
            st.plotly_chart(fig, use_container_width=True) 

        #**********************************************************棒グラフ　今期前期
        def make_bar_nowlast(self, lists_now, lists_last, x_list):
            #可視化
            #グラフを描くときの土台となるオブジェクト
            fig = go.Figure()
            #今期のグラフの追加
            
            for (val_list, name) in zip([lists_now, lists_last], ['今期', '前期']) :
                fig.add_trace(
                    go.Bar(
                        x=x_list,
                        y=val_list,  
                        text=[round(val/10000) if val >= 10000 else int(val) for val in val_list],
                        textposition="outside", 
                        name=name)
                )
            #レイアウト設定     
            fig.update_layout(
                showlegend=True #凡例表示
            )
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
            st.plotly_chart(fig, use_container_width=True) 

        #**********************************************************棒グラフ　今期前期 小数
        def make_bar_nowlast_float(self, lists_now, lists_last, x_list):
            #可視化
            #グラフを描くときの土台となるオブジェクト
            fig = go.Figure()
            #今期のグラフの追加
            
            for (val_list, name) in zip([lists_now, lists_last], ['今期', '前期']) :
                fig.add_trace(
                    go.Bar(
                        x=x_list,
                        y=val_list,  
                        text=[round(val, 2) for val in val_list],
                        textposition="outside", 
                        name=name)
                )
            #レイアウト設定     
            fig.update_layout(
                showlegend=True #凡例表示
            )
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
            st.plotly_chart(fig, use_container_width=True) 
        
        #*************************************************************棒グラフ　横 基準線あり
         #可視化
        def make_bar_h(self, val_list, label_list, name, title, line_val, height):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=val_list,
                y=label_list,
                marker_color='#87cefa',
                textfont={'color': '#696969'},
                name=name)
                )
            fig.update_traces(
                textposition='outside',
                texttemplate='%{x:0.2f}',
                orientation='h'
                )
            # 基準線の追加
            fig.add_shape(
                type="line",
                x0=line_val,  # 基準線の開始位置 (x座標)
                x1=line_val,  # 基準線の終了位置 (x座標)
                y0=label_list[0],  # 基準線の開始位置 (y座標)
                y1=label_list[-1],  # 基準線の終了位置 (y座標)
                line=dict(
                    color="red",
                    width=2,
                    dash="dash"  # 破線を使用する場合は "dash" を指定
        )
    )
            fig.update_layout(
                title=title,
                width=500,
                height=height,
                plot_bgcolor='white'
                )
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
            st.plotly_chart(fig, use_container_width=True) 
        
        #*************************************************************棒グラフ　横　基準線なし
         #可視化
        def make_bar_h_nonline(self, val_list, label_list, name, title, height):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=val_list,
                y=label_list,
                marker_color='#87cefa',
                textfont={'color': '#696969'},
                name=name)
                )
            fig.update_traces(
                textposition='outside',
                texttemplate='%{x}',
                orientation='h'
                )

            fig.update_layout(
                title=title,
                width=500,
                height=height,
                plot_bgcolor='white'
                )
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
            st.plotly_chart(fig, use_container_width=True) 

        #**********************************************************折れ線
        def make_line(self, df_list, name_list, x_list):

            #グラフを描くときの土台となるオブジェクト
            fig = go.Figure()
            #今期のグラフの追加

            for (df, name) in zip(df_list, name_list):

                fig.add_trace(
                go.Scatter(
                    x=x_list, #strにしないと順番が崩れる
                    y=df,
                    mode = 'lines+markers+text', #値表示
                    text=[round(val/10000) if val >= 10000 else int(val) for val in df],
                    textposition="top center", 
                    name=name)
                    )  

            #レイアウト設定     
            fig.update_layout(
                showlegend=True #凡例表示
            )
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
            st.plotly_chart(fig, use_container_width=True) 
        
        #**********************************************折れ線　non_xlist    
        def make_line_nonXlist(self, df_list, name_list):
            #グラフを描くときの土台となるオブジェクト
            fig = go.Figure()
            #今期のグラフの追加

            for (df, name) in zip(df_list, name_list):

                fig.add_trace(
                go.Scatter(
                    x=['10月', '11月', '12月', '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月'], #strにしないと順番が崩れる
                    y=df,
                    mode = 'lines+markers+text', #値表示
                    text=[round(val/10000) if val >= 10000 else int(val) for val in df],
                    textposition="top center", 
                    name=name)
                    )  

            #レイアウト設定     
            fig.update_layout(
                showlegend=True #凡例表示
            )
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
            st.plotly_chart(fig, use_container_width=True) 
            
        #***************************************************************円
        def make_pie(self, vals, labels):

            # st.write(f'{option_category} 構成比(今期)')
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=vals
                        )])
            fig.update_layout(
                showlegend=True, #凡例表示
                height=290,
                margin={'l': 20, 'r': 60, 't': 0, 'b': 0},
                )
            fig.update_traces(textposition='inside', textinfo='label+percent') 
            #inside グラフ上にテキスト表示
            st.plotly_chart(fig, use_container_width=True) 
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅

        #***************************************************************箱ひげ/今期前期
        def make_box(self, list_now, list_last, name_list):
            fig = go.Figure()
            fig.add_trace(go.Box(y=list_now,
                                name=name_list[0]))
            fig.add_trace(go.Box(y=list_last,
                                name=name_list[1]))
            fig.update_traces(boxpoints='all', jitter=0.3) 
            #散布図　jitter=0.3として散布図の幅(広がり方)を指定
            st.plotly_chart(fig, use_container_width=True) 
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅
        
        #***************************************************************箱ひげ/今期のみ
        def make_box_now(self, list_now, name):
            fig = go.Figure()
            fig.add_trace(go.Box(y=list_now,
                                name=name))
            fig.update_traces(boxpoints='all', jitter=0.3) 
            #散布図　jitter=0.3として散布図の幅(広がり方)を指定
            st.plotly_chart(fig, use_container_width=True) 
            #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅



#*******************************************アイテムの深堀
#******************データの絞込み
def fukabori(df_now, df_now2, graph):
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
            key='fukabori_sl'
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
            graph.make_bar_h_nonline(s_item[-20:], s_item.index[-20:], '数量', '売れ筋組み合わせ 塗色／張地', 800)

        #********************購入している得意先
        s_sum = df_select.groupby('得意先名')['数量'].sum()

        s_sum = s_sum.sort_values(ascending=True)
        graph.make_bar_h_nonline(s_sum[-20:], s_sum.index[-20:], '数量', '得意先別数量', 800)

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

#*******************************************アイテムの深堀　品番渡し
#******************データの絞込み
def fukabori2(hinban, df_now, df_now2, selected_base, graph):

    df_select = df_now2[df_now2['品番'] == hinban]

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
        graph.make_bar_h_nonline(s_item[-20:], s_item.index[-20:], '数量', '売れ筋組み合わせ 塗色／張地', 800)

    #********************購入している得意先
    st.markdown('##### 得意先別数量')
    s_sum = df_select.groupby('得意先名')['数量'].sum()

    s_sum = s_sum.sort_values(ascending=True)
    graph.make_bar_h_nonline(s_sum[-20:], s_sum.index[-20:], '数量', '得意先別数量', 800)

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
    st.markdown('#### 得意先別分析')
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
    st.markdown('##### 同時に買われているアイテムを見る')
    df_concat = pd.DataFrame()
    for num in df_cust2['伝票番号2']:
        df = df_now[df_now['伝票番号2'] == num]
        df_concat = pd.concat([df_concat, df], join='outer')
    
    if len(df_concat) == 0:
            st.write('購入実績がありません')
    
    with st.expander('明細', expanded=False):
        if len(df_concat) == 0:
            st.write('購入実績がありません')
        else:
            col_list = ['得意先名', '商　品　名', '数量', '伝票番号2']
            st.table(df_concat[col_list])
        
    #箱ひげ
    df_hinban_now = df_now2[df_now2['品番']==hinban]
    s_cust_now = df_hinban_now.groupby('得意先名')[selected_base].sum()

    #可視化
    st.markdown('##### 数量の分布/箱ひげ')
    st.write('得意先数')
    st.write(len(s_cust_now))
    graph.make_box_now(s_cust_now, '今期')

    #月次推移
    st.markdown('##### 月次推移')
    df_suii = df_now2[df_now2['品番']==hinban]
    df_suii = df_suii[df_suii['得意先名']==selected_cust]
    s_suii =df_suii.groupby('受注年月')[selected_base].sum()

    #可視化
    graph.make_line([s_suii], ['今期'], s_suii.index)

    #試算
    
    st.markdown('##### 年間販売予測')

    data_span =  (df_now['受注日'].max() - df_now['受注日'].min()).days
    #days属性を使用してTimedeltaオブジェクトの日数を取得
    span_rate = 365 / data_span
    
    if len(s_cust_now) == 0:
        st.write('購入実績がありません')
    else:
        med = s_cust_now.median()*span_rate
        st.write(f'■ 中央値: {round(med)}')
        q90 = s_cust_now.quantile(0.9)*span_rate
        st.write(f'■ 上位90%: {round(q90)}')
        q100 = s_cust_now.max()*span_rate
        st.write(f'■ 最大値: {round(q100)}')
    
    link = '[アソシエーション分析](https://cocosan1-association-fullhinban-cmy4cf.streamlit.app/)'
    st.markdown(link, unsafe_allow_html=True)

    return selected_cust

######################################################################深堀　ピンポイントの内容
def fukabori3(df_now, df_last, selected_base, selected_cate,  graph):
    st.markdown('#### ピンポイント品番分析/年間予測')

    #年換算
    date_min = df_now['受注日'].min()
    date_max = df_now['受注日'].max()
    date_span = (date_max - date_min).days
    date_rate = 365 / date_span

    df_now_year = df_now.copy()
    df_last_year = df_last.copy()
    df_now_year[selected_base] = df_now[selected_base] * date_rate
    df_last_year[selected_base] = df_last[selected_base] * date_rate
    df_now_year[selected_base] = df_now_year[selected_base].astype('int')
    df_last_year[selected_base] = df_last_year[selected_base].astype('int')

    #前処理
    df_now2, df_last2 = pre_processing(df_now, df_last, selected_base, selected_cate)

    s_now2g = df_now2.groupby('品番')[selected_base].sum()
    s_last2g = df_last2.groupby('品番')[selected_base].sum()
    
    df_now2['受注年月'] = df_now2['受注日'].dt.strftime("%Y-%m")
    df_now2['受注年月'] = pd.to_datetime(df_now2['受注年月'])
    
    st.write(f'【今期数量】{s_now2g.sum()} 【前期数量】{s_last2g.sum()} \
             【対前年比】{s_now2g.sum() / s_last2g.sum():.2f}')

    hinban_list = sorted(list(df_now2['商品コード2'].unique()))
    hinbans = st.multiselect(
        '品番の選択/複数可',
        hinban_list,
        key='hinbans'
    )
    s_now = s_now2g[hinbans]
    #前年の売上がない商品の対処
    for hinban in hinbans:
        if hinban in s_last2g.index:
            continue
        else:
            s_last2g[hinban] = 0

    s_last = s_last2g[hinbans]

    #一覧の作成
    #数量データ分布一覧/itemベース
    items = []
    cnts = []
    quan25s = []
    medis = []
    quan75s = []
    quan90s = []
    maxs = []
    span2575s = []
    den_cnts = []
    den_rates = []
    for item in hinbans:
        items.append(item)
        df_item = df_now2[df_now2['品番']==item]
        s_cust = df_item.groupby('得意先名')[selected_base].sum()
        
        cnt = s_cust.count()
        quan25 = round(s_cust.quantile(0.25), 1)
        medi = s_cust.median()
        quan75 = round(s_cust.quantile(0.75), 1)
        quan90 = round(s_cust.quantile(0.9), 1)
        max_num = s_cust.max()
        span2575 = quan75 - quan25
        den_cnt = df_item['伝票番号2'].nunique()
        den_rate = round(den_cnt / cnt, 1)

        cnts.append(cnt)
        quan25s.append(quan25)
        medis.append(medi)
        quan75s.append(quan75)
        quan90s.append(quan90)
        maxs.append(max_num)
        span2575s.append(span2575)
        den_cnts.append(den_cnt)
        den_rates.append(den_rate)

    df_calc = pd.DataFrame(list(zip(cnts, quan25s, medis, quan75s, quan90s, maxs, span2575s, den_cnts, \
                                    den_rates)), \
                    columns=['得意先数', '第2四分位', '中央値', '第3四分位', '上位10%', '最大値', 'span2575', \
                             '伝票数', '伝票数/得意先数'], index=items)
    with st.expander('df_calc', expanded=False):
        st.dataframe(df_calc)
    
    st.markdown('##### 得意先数')
    graph.make_bar(df_calc['得意先数'], df_calc.index)

    st.markdown('##### 伝票数/得意先数')
    graph.make_bar(df_calc['伝票数/得意先数'], df_calc.index)
    st.write(df_calc['伝票数/得意先数'])
    
    st.markdown('##### 前年比')
    graph.make_bar_nowlast(s_now, s_last, s_now.index)

    rates = []
    indxs = []
    for (now, last, indx) in zip(s_now, s_last, s_now.index):
        if last == 0:
            rate = 0
        else:
            rate = round(now / last, 2)

        rates.append(rate)
        indxs.append(indx)
    s_rate = pd.Series(rates, index=indxs)
    st.write(s_rate)


    #偏差値
    #標準偏差
    std_now = s_now2g.std(ddof=0)
    std_last = s_last2g.std(ddof=0)
    #平均
    mean_now = s_now2g.mean()
    mean_last = s_last2g.mean()

    df_now_temp = pd.DataFrame(s_now2g)
    df_last_temp = pd.DataFrame(s_last2g)

    df_nowlast = df_now_temp.merge(df_last_temp, left_index=True, right_index=True, how='left')
    df_nowlast = df_nowlast.rename(columns={f'{selected_base}_x': '今期', f'{selected_base}_y': '前期'})
   
    df_nowlast = df_nowlast.fillna(0)
    df_nowlast['偏差値/今期'] = df_nowlast['今期'].map(lambda x: (x - mean_now) / std_now * 10 + 50)
    df_nowlast['偏差値/前期'] = df_nowlast['前期'].map(lambda x: (x - mean_last) / std_last * 10 + 50)
    df_nowlast['偏差値/今期'] = df_nowlast['偏差値/今期'].map(lambda x: round(x, 2))
    df_nowlast['偏差値/前期'] = df_nowlast['偏差値/前期'].map(lambda x: round(x, 2))
    df_nowlast['偏差値/推移'] = df_nowlast['偏差値/今期'] - df_nowlast['偏差値/前期']

    st.markdown('##### 偏差値')
    df_selected = df_nowlast.loc[hinbans]
    graph.make_bar_nowlast(df_selected['偏差値/今期'], df_selected['偏差値/前期'], df_selected.index)

    col1, col2 = st.columns(2)
    with col1:
        df_rank_now = df_nowlast['今期'].rank(ascending=False)
        df_rank_now = df_rank_now.loc[hinbans]
        st.write('順位')
        st.write(df_rank_now)
        st.write(f'item: {len(s_now2g)}')
    
    with col2:
        df_rank_last = df_nowlast['前期'].rank(ascending=False)
        df_rank_last = df_rank_last.loc[hinbans]
        st.write('順位')
        st.write(df_rank_last)
        st.write(f'item: {len(s_last2g)}')


    # #可視化
    st.markdown('##### 数量の分布/箱ひげ')

    df_hinban_now = df_now2[df_now2['品番'].isin(hinbans)]

    fig = go.Figure()
    for hinban in hinbans:
        df = df_hinban_now[df_hinban_now['品番']==hinban]
        s_now = df.groupby('得意先名')[selected_base].sum()

        fig.add_trace(go.Box(y=s_now, name=hinban))
        fig.update_traces(boxpoints='all', jitter=0.3) 
        #散布図　jitter=0.3として散布図の幅(広がり方)を指定

    # fig.update_traces(boxpoints='all', jitter=0.3)
    fig.update_layout(showlegend=False) 

    st.plotly_chart(fig, use_container_width=True) 
    #plotly_chart plotlyを使ってグラグ描画　グラフの幅が列の幅



    #深堀
    st.markdown('### 深堀り/売上調整なし')
    hinban2 = st.selectbox(
        '品番の選択',
        hinbans,
        key='hinban2'
    )


        #深堀関数
    fukabori2(hinban2, df_now, df_now2, selected_base, graph)

#************************************************************相関
#index商品分類/col得意先/val売上
@st.cache_data(ttl=datetime.timedelta(hours=1))
def make_data_corr(df):

    #必要なカラムだけに絞る
    df = df[['得意先名', '金額', '商品コード2']]

    #得意先毎に集計
    custs = df['得意先名'].unique()
    items = df['商品コード2'].unique()
    df_calc = pd.DataFrame(index=items)
    for cust in custs:
        vals= []
        df_cust = df[df['得意先名']==cust]
        for item in items:
            df_cust2 = df_cust[df_cust['商品コード2']==item]
            if len(df_cust2) == 0:
                vals.append(0)
            else:
                val = df_cust2['金額'].sum()
                vals.append(val)
        df_temp = pd.DataFrame(vals, columns=[cust], index=items)
        df_calc = df_calc.merge(df_temp, left_index=True, right_index=True, how='outer') 

    df_calc.loc['sales'] = df_calc.sum()
    df_calc = df_calc.T
    
    return df_calc
