import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.fft import fft
from Viz_toolbox import Preprocessing as p
from dash import dash_table
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.stats import boxcox, yeojohnson
from statsmodels.graphics.gofplots import qqplot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash('Global Terrorism', external_stylesheets = external_stylesheets)

terrorism_df = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
description_df = terrorism_df.describe()
description_df.reset_index(inplace=True)
data_clean, description_cleaned_df = p.missing_data(terrorism_df)
description_cleaned_df.reset_index(inplace=True)

app.layout = html.Div([
    html.H1('Global Terrorism', style={'text-align':'center'}),
    dcc.Tabs(id='tabs', value='tabs', children=[
        dcc.Tab(label='Overview', value='tabA',
            children=[
                dcc.Tabs(id="subtabA",  value="subtabs",children = [
                dcc.Tab(label='Head of the data', value="subtabA1"),
                dcc.Tab(label='Data Statistics', value="subtabA2")])]),
        dcc.Tab(label='Exploratory Data Analysis', value='tabB', children=[
            dcc.Tabs(id="subtabB",  value="subtab", children = [
                dcc.Tab(label='Outlier detection & removal', value="subtabB1"),
                dcc.Tab(label='Principal Component Analysis (PCA)', value="subtabB2"),
                dcc.Tab(label='Normality Test', value="subtabB3"),
                dcc.Tab(label='Pearson correlation coefficient heatmap', value="subtabB4")])]),
        dcc.Tab(label='Trend in Terrorist Activities Over the Period of Time', value='tabC', children=[
                dcc.Tabs(id="subtabC",  value="subtab", children = [
                dcc.Tab(label='Number Of Terrorist Activities Each Year', value="subtabC1"),
                dcc.Tab(label='Trend in Terrorist Activities Based on Region', value="subtabC2"),
                dcc.Tab(label='Number Of Fatalities and Injuries In Each Region', value="subtabC3"),
                dcc.Tab(label='Number Of Fatalities and Injuries for Specific Attacktype', value="subtabC4"),
                dcc.Tab(label='Number Of Fatalities and Injuries for Specific Weapontype', value="subtabC5")])]),
        dcc.Tab(label='Reasons for the Attacks of Terror', value='tabD', children=[
            dcc.Tabs(id="subtabD", value="subtab", children=[
                dcc.Tab(label='Number Of Terrorist Activities WRT Reasons of Terror Each Year', value="subtabD1"),
                dcc.Tab(label='Number Of Terrorist Activities WRT Reasons of Terror Each Year for Specific AttackType', value="subtabD2"),
                dcc.Tab(label='Number Of Terrorist Activities WRT Reasons of Terror Each Year for Specific Weapontype', value="subtabD3")])]),
        ]),
html.Div(id='tabs-content')
])

question1_layout = html.Div([
    html.P('Select the dataset:'),
    dcc.RadioItems( id = 'input1',
       options=['Original', 'Cleaned'],
       value='Original', inline = True),
    html.Br(),
    html.Div(id = 'my_out-1')
])

question11_layout = html.Div([
    html.P('Select the dataset:'),
    dcc.RadioItems( id = 'input11',
       options=['Original', 'Cleaned'],
       value='Original', inline = True),
    html.Br(),
    html.Div(id = 'my_out-11')
])

question2_layout = html.Div([
    html.P('Select the dataset:'),
    dcc.RadioItems(id='input2',
                   options=['Dataset with Outliers', 'Dataset without Outliers'],
                   value='Dataset with Outliers', inline=True),
    html.Br(),
    html.P('Select the column :'),
    dcc.Dropdown(id = 'drop2',
                 options = [
                    {'label':'Number of Kills','value':'nkill'},
                    {'label':'Number of Wounds','value':'nwound'},
                 ], value= 'nkill'),
    html.Br(),
    dcc.Graph(id = 'my_out-2')
])

question21_layout = html.Div([
    html.P('Mention the number of components:'),
    dcc.Input(id = 'input21', type = 'number',value = 5, max = 24),
    html.Br(),
    # dcc.Graph(id = 'my_out-2')
    html.Div([
        html.H4('cumulative explained variance v/s number of components'),
        dcc.Graph(id = 'my_out-21'),],
        style = {"width": '50%', "margin": 0, 'display': 'inline-block'}),
    html.Div([
        html.H4('Correlation Coefficent between featues - Reduced feature space1'),
        dcc.Graph(id = 'my_out-211'),],
        style = {"width": '50%', "margin": 0, 'display': 'inline-block'})
])

question22_layout = html.Div([
    html.P('Select the transformation type :'),
    dcc.RadioItems(id='input22',
                   options=['Square-Root Transformation', 'Yeo-Johnson Transformation'],
                   value='Yeo-Johnson Transformation', inline=True),
    html.Br(),
    html.P('Select the column :'),
    dcc.Dropdown(id = 'drop22',
                 options = [
                    {'label':'Number of Kills','value':'nkill'},
                    {'label':'Number of Wounds','value':'nwound'},
                 ], value= 'nkill'),
    html.Br(),
    # dcc.Graph(id = 'my_out-2')
    html.Div([
        html.H4('Data Before Transformation'),
        dcc.Graph(id = 'my_out-22'),],
        style = {"width": '50%', "margin": 0, 'display': 'inline-block'}),
    html.Div([
        html.H4('Data After Transformation'),
        dcc.Graph(id = 'my_out-221'),],
        style = {"width": '50%', "margin": 0, 'display': 'inline-block'})
])

question23_layout = html.Div([
    html.P('Select the features for correlation '),
    dcc.Checklist(id='cc23',
                 options=[
                     {'label': 'Year', 'value': 'iyear'},
                     {'label': 'Month', 'value': 'imonth'},
                     {'label': 'Day', 'value': 'iday'},
                     {'label': 'Extended Incident', 'value': 'extended'},
                     {'label': 'Country', 'value': 'country'},
                     {'label': 'Region', 'value': 'region'},
                     {'label': 'Vicinity', 'value': 'vicinity'},
                     {'label': 'Political, Economic, Religious, or Social Goal.', 'value': 'crit1'},
                     {'label': 'Intention to coerce, intimidate, or publicize to a larger audience', 'value': 'crit2'},
                     {'label': 'Outside  International Humanitarian Law', 'value': 'crit3'},
                     {'label': 'Successful Attack (Target Variable)', 'value': 'success'},
                     {'label': 'Attack Type', 'value': 'attacktype1'},
                     {'label': 'Target/Victim Type', 'value': 'targtype1'},
                     {'label': 'Unaffiliated Individual', 'value': 'individual'},
                     {'label': 'Weapon Type', 'value': 'weaptype1'},
                     {'label': 'Number of Kills', 'value': 'nkill'},
                     {'label': 'Number of Wounds', 'value': 'nwound'},
                     {'label': 'Property Damage', 'value': 'property'}
                 ], inline =True),
    html.Br(),
    # html.Div(id = 'my_out_23')
    dcc.Graph(id = 'my_out_23')
])

question3_layout = html.Div([
    html.H4('Number Of Terrorist Activities Each Year', style={'text-align':'center'}),
    dcc.RadioItems(id='input-3',
                   options=['Terrorist Activities Each Year'],
                   value='Terrorist Activities Each Year', inline=True),
    html.Br(),
    dcc.Graph(id = 'my_out_3')
])

question31_layout = html.Div([
    html.H4('Number Of Terrorist Activities Each Year', style={'text-align':'center'}),
    dcc.RadioItems(id='input-31',
                   options=['Terrorist Activities Each Year'],
                   value='Terrorist Activities Each Year', inline=True),
    html.Br(),
    dcc.Graph(id = 'my_out_31')
])

question32_layout = html.Div([
    html.H4('Number Of Fatalities and Injuries In Each Region', style={'text-align':'center'}),
    dcc.Graph(id='my_out_32'),
    dcc.Slider(
        data_clean['iyear'].min(),
        data_clean['iyear'].max(),
        step=None,
        value=data_clean['iyear'].min(),
        marks={str(year): str(year) for year in data_clean['iyear'].unique()},
        id='input-32'
    )
])

question33_layout = html.Div([
    html.H4('Number Of Fatalities and Injuries for Specific Attacktype', style={'text-align':'center'}),
    dcc.Slider(
        data_clean['iyear'].min(),
        data_clean['iyear'].max(),
        step=None,
        value=data_clean['iyear'].min(),
        marks={str(year): str(year) for year in data_clean['iyear'].unique()},
        id='input-33'
    ),
    html.Div([
        html.H4('Number of Injuries in Year based on AttackType', style={'text-align':'center'}),
        dcc.Graph(id = 'my_out_331'),],
        style = {"width": '50%', "margin": 0, 'display': 'inline-block'}),
    html.Div([
        html.H4('Number of Fatalities in Year based on AttackType', style={'text-align':'center'}),
        dcc.Graph(id = 'my_out_332'),],
        style = {"width": '50%', "margin": 0, 'display': 'inline-block'})
    # dcc.Graph(id='my_out_32'),
])

question34_layout = html.Div([
    html.H4('Number Of Fatalities and Injuries for Specific WeaponType', style={'text-align':'center'}),
    dcc.Slider(
        data_clean['iyear'].min(),
        data_clean['iyear'].max(),
        step=None,
        value=data_clean['iyear'].min(),
        marks={str(year): str(year) for year in data_clean['iyear'].unique()},
        id='input-34'
    ),
    html.Div([
        html.H4('Number of Injuries in Year based on WeaponType', style={'text-align':'center'}),
        dcc.Graph(id = 'my_out_341'),],
        style = {"width": '50%', "margin": 0, 'display': 'inline-block'}),
    html.Div([
        html.H4('Number of Fatalities in Year based on WeaponType', style={'text-align':'center'}),
        dcc.Graph(id = 'my_out_342'),],
        style = {"width": '50%', "margin": 0, 'display': 'inline-block'})
    # dcc.Graph(id='my_out_32'),
])

question4_layout = html.Div([
    html.P('Select the reason for terrorism attack :'),
    dcc.Dropdown(id = 'drop4',
                 options = [
                     {'label': 'Political, Economic, Religious, or Social Goal.', 'value': 'crit1'},
                     {'label': 'Intention to coerce, intimidate, or publicize to a larger audience', 'value': 'crit2'},
                     {'label': 'Outside  International Humanitarian Law', 'value': 'crit3'}
                 ], value= 'crit1'),
    html.Br(),
    dcc.Graph(id = 'my_out_4')

])


question41_layout = html.Div([
    html.P('Select the reason for terrorism attack :'),
    dcc.Dropdown(id = 'drop41',
                 options = [
                     {'label': 'Political, Economic, Religious, or Social Goal.', 'value': 'crit1'},
                     {'label': 'Intention to coerce, intimidate, or publicize to a larger audience', 'value': 'crit2'},
                     {'label': 'Outside  International Humanitarian Law', 'value': 'crit3'}
                 ], value= 'crit1'),
    html.Br(),
    dcc.Graph(id = 'my_out_41')

])

question42_layout = html.Div([
    html.P('Select the reason for terrorism attack :'),
    dcc.Dropdown(id = 'drop42',
                 options = [
                     {'label': 'Political, Economic, Religious, or Social Goal.', 'value': 'crit1'},
                     {'label': 'Intention to coerce, intimidate, or publicize to a larger audience', 'value': 'crit2'},
                     {'label': 'Outside  International Humanitarian Law', 'value': 'crit3'}
                 ], value= 'crit1'),
    html.Br(),
    dcc.Graph(id = 'my_out_42')

])

@app.callback(
    Output(component_id = 'tabs-content',component_property='children'),
    [Input(component_id = 'tabs',component_property='value'),
     Input(component_id = 'subtabA',component_property='value'),
     Input(component_id='subtabB', component_property='value'),
     Input(component_id='subtabC', component_property='value'),
     Input(component_id='subtabD', component_property='value')]
)


def update_layout(ques, subtabA, subtabB, subtabC, subtabD):
    if ques == 'tabA':
        if subtabA == 'subtabA1':
            return question1_layout
        if subtabA == 'subtabA2':
            return question11_layout
    elif ques == 'tabB':
        if subtabB == 'subtabB1':
            return question2_layout
        elif subtabB == 'subtabB2':
            return question21_layout
        elif subtabB == 'subtabB3':
            return question22_layout
        elif subtabB == 'subtabB4':
            return question23_layout
    elif ques == 'tabC':
        if subtabC == 'subtabC1':
            return question3_layout
        elif subtabC == 'subtabC2':
            return question31_layout
        elif subtabC == 'subtabC3':
            return question32_layout
        elif subtabC == 'subtabC4':
            return question33_layout
        elif subtabC == 'subtabC5':
            return question34_layout
    elif ques == 'tabD':
        if subtabD == 'subtabD1':
            return question4_layout
        elif subtabD == 'subtabD2':
            return question41_layout
        elif subtabD == 'subtabD3':
            return question42_layout
        # elif subtabC == 'subtabC4':
        #     return question33_layout
        # elif subtabC == 'subtabC5':
        #     return question34_layout




@app.callback(
    dash.dependencies.Output(component_id = 'my_out-1',component_property='children'),
    [dash.dependencies.Input(component_id = 'input1',component_property='value')]
)

def update_head(input1):
    if input1 == 'Original':
        return html.Div([ html.H3('Orginal Dataset', style = {'textAlign': 'center'}),
                           html.Br(),
                          html.P('The shape of original dataset is  (181691, 135)'),
                          html.Br(),
                            dash_table.DataTable(
                                data = terrorism_df[:50].to_dict('records'),
                                columns=[{'id': c, 'name': c} for c in terrorism_df[:50].columns],
                            )])
    if input1 == 'Cleaned':
        return html.Div([ html.H3('Cleaned Dataset', style = {'textAlign': 'center'}),
                           html.Br(),
                          html.P('The shape of cleaned dataset is  (181691, 31)'),
                          html.Br(),
                            dash_table.DataTable(
                                data = data_clean[:50].to_dict('records'),
                                columns=[{'id': c, 'name': c} for c in data_clean[:50].columns],
                            )])

@app.callback(
    dash.dependencies.Output(component_id = 'my_out-11',component_property='children'),
    [dash.dependencies.Input(component_id = 'input11',component_property='value')]
)

def update_stats(input11):
    if input11 == 'Original':
        return html.Div([ html.H3('Statistics of Orginal Dataset', style = {'textAlign': 'center'}),
                           html.Br(),
                            dash_table.DataTable(
                                data = description_df.to_dict('records'),
                                columns=[{'id': c, 'name': c} for c in description_df.columns],
                            )])
    if input11 == 'Cleaned':
        return html.Div([ html.H3('Statistics of Cleaned Dataset', style = {'textAlign': 'center'}),
                           html.Br(),
                            dash_table.DataTable(
                                data = description_cleaned_df.to_dict('records'),
                                columns=[{'id': c, 'name': c} for c in description_cleaned_df.columns],
                            )])

@app.callback(
    dash.dependencies.Output(component_id = 'my_out-2',component_property='figure'),
    [dash.dependencies.Input(component_id = 'input2',component_property='value'),
     dash.dependencies.Input(component_id = 'drop2',component_property='value')]
)

def outlier_detection(input2, drop2):
    if input2 == 'Dataset with Outliers':
        df = data_clean [:3000]
        fig = px.box(df, y = drop2, title = f'Box plot of {drop2} column')
        # fig.show()
        return fig
    elif input2 == 'Dataset without Outliers':
        df = data_clean[:3000]
        Q1 = np.percentile(df[drop2], 25,
                           interpolation='midpoint')

        Q3 = np.percentile(df[drop2], 75,
                           interpolation='midpoint')
        IQR = Q3 - Q1
        upper = np.where(df[drop2] >= (Q3 + 1.5 * IQR))
        # Lower bound
        lower = np.where(df[drop2] <= (Q1 - 1.5 * IQR))
        df.drop(upper[0], inplace=True)
        df.drop(lower[0], inplace=True)
        fig = px.box(df, y = drop2, title = f'Box plot after outlier removal in {drop2} column')
        # fig.show()
        return fig

@app.callback(
    dash.dependencies.Output(component_id = 'my_out-21',component_property='figure'),
    dash.dependencies.Output(component_id = 'my_out-211',component_property='figure'),
    [dash.dependencies.Input(component_id = 'input21',component_property='value')]
)

def pca_analysis(input21):
    df = data_clean
    X = data_clean.drop(['success','country_txt','region_txt','attacktype1_txt','targtype1_txt', 'weaptype1_txt'], axis=1)
    # PCA Analysis
    label_encoder = preprocessing.LabelEncoder()
    X['gname'] = label_encoder.fit_transform(X['gname'])
    X = StandardScaler().fit_transform(X)
    # pca = PCA(n_components='mle', svd_solver='full')
    pca = PCA(n_components = input21 , svd_solver='full')
    pca.fit(X)
    X_PCA = pca.transform(X)

    x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1)

    fig = px.line( x = x, y = np.cumsum(pca.explained_variance_ratio_) )
    fig.update_yaxes(title= 'Variance')
    fig.update_xaxes(title='Columns')

    a, b = X_PCA.shape
    column = []

    for i in range(b):
        column.append(f'Principal Col {i}')

    df_PCA = pd.DataFrame(data=X_PCA, columns=column)
    reduced = df_PCA.corr(method='pearson')
    fig1 = px.imshow(reduced, text_auto=True)

    return fig,fig1

@app.callback(
    dash.dependencies.Output(component_id = 'my_out-22',component_property='figure'),
    dash.dependencies.Output(component_id = 'my_out-221',component_property='figure'),
    [dash.dependencies.Input(component_id = 'input22',component_property='value'),
     dash.dependencies.Input(component_id = 'drop22',component_property='value')]
)

def normality(input22, drop22):

    df = data_clean[drop22]
    df = df[:50000]

    if input22 == 'Square-Root Transformation':
        df_norm = df**(1/2)
    elif input22 == 'Box-Cox Transformation':
        df_norm, _ = boxcox(df)
    elif input22 == 'Yeo-Johnson Transformation':
        df_norm, _ = yeojohnson(df)

    qqplot_data = qqplot(df, line='s').gca().lines

    fig = go.Figure()

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    })

    fig['layout'].update({
        'title': 'Quantile-Quantile Plot',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
    })

    # fig.show()

    qqplot_data1 = qqplot(df_norm, line='s').gca().lines

    fig1 = go.Figure()

    fig1.add_trace({
        'type': 'scatter',
        'x': qqplot_data1[0].get_xdata(),
        'y': qqplot_data1[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    })

    fig1.add_trace({
        'type': 'scatter',
        'x': qqplot_data1[1].get_xdata(),
        'y': qqplot_data1[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    })

    fig1['layout'].update({
        'title': 'Quantile-Quantile Plot',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
    })

    return fig, fig1

@app.callback(
    dash.dependencies.Output(component_id = 'my_out_23',component_property='figure'),
    # dash.dependencies.Output(component_id = 'my_out-221',component_property='figure'),
    [dash.dependencies.Input(component_id = 'cc23',component_property='value')]
)

def p_corr(cc23):

    df = data_clean[cc23]
    reduced = df.corr(method='pearson')
    fig1 = px.imshow(reduced, text_auto=True, title = 'Correlation Matrix')
    # fig1.show()
    fig1['layout'].update({
        'height': 700,
    })

    return fig1

@app.callback(
    dash.dependencies.Output(component_id = 'my_out_3',component_property='figure'),
    [dash.dependencies.Input(component_id = 'input-3',component_property='value')]
)

def terror_over_time(s):

    df = data_clean.groupby('iyear').success.sum()

    fig = px.bar(x = df.index, y = df, color_continuous_scale = 'viridis')
    fig['layout'].update({
        'title': 'Terrorism over the Years',
        'xaxis': {
            'title': 'Years',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Number of Attacks'
        }
    })

    return fig

@app.callback(
    dash.dependencies.Output(component_id = 'my_out_31',component_property='figure'),
    [dash.dependencies.Input(component_id = 'input-31',component_property='value')]
)

def terror_over_time_region_line(s):

    df1 = data_clean[['iyear', 'success', 'region_txt']].groupby(['iyear', 'region_txt'], as_index=False).sum()

    fig = px.line(df1, x="iyear", y="success", color = "region_txt", title='Number Of Terrorist Activities Each Year')

    # fig.show()

    return fig

@app.callback(
    dash.dependencies.Output(component_id = 'my_out_32',component_property='figure'),
    [dash.dependencies.Input(component_id = 'input-32',component_property='value')]
)

def terror_over_time_region_bar(selected_year):

    # df = data_clean.groupby('iyear','region').success.sum()

    filtered_df = data_clean[data_clean.iyear == selected_year]

    fig = px.scatter(filtered_df, x="nkill", y="nwound",
                     color="region_txt", hover_name="country_txt", size = "nwound",
                     log_x=True, size_max=55)

    return fig

@app.callback(
        dash.dependencies.Output(component_id='my_out_331', component_property='figure'),
        dash.dependencies.Output(component_id='my_out_332', component_property='figure'),
        [dash.dependencies.Input(component_id='input-33', component_property='value')]
)

def terror_over_region_attacktype(selected_year):

    # df = data_clean.groupby('iyear','region').success.sum()

    filtered_df = data_clean[data_clean.iyear == selected_year]

    fig = px.pie(filtered_df, values='nwound', names='attacktype1_txt')

    fig1 = px.pie(filtered_df, values='nkill', names='attacktype1_txt')

    return fig, fig1

@app.callback(
        dash.dependencies.Output(component_id='my_out_341', component_property='figure'),
        dash.dependencies.Output(component_id='my_out_342', component_property='figure'),
        [dash.dependencies.Input(component_id='input-34', component_property='value')]
)

def terror_over_region_attacktype(selected_year):

    # df = data_clean.groupby('iyear','region').success.sum()

    filtered_df = data_clean[data_clean.iyear == selected_year]

    fig = px.pie(filtered_df, values='nwound', names='weaptype1_txt')

    fig1 = px.pie(filtered_df, values='nkill', names='weaptype1_txt')

    return fig, fig1

@app.callback(
        dash.dependencies.Output(component_id='my_out_4', component_property='figure'),
        [dash.dependencies.Input(component_id='drop4', component_property='value')]
)

def terror_over_time_reasons(cr):

    df1 = data_clean[['iyear', 'success', cr]].groupby(['iyear', cr], as_index=False).sum()
    df2 = df1[df1[cr] == 1]

    fig = px.line(df2, x="iyear", y="success", color = cr , title='Number Of Terrorist Activities WRT Reasons of Terror Each Year')

    return fig

@app.callback(
        dash.dependencies.Output(component_id='my_out_41', component_property='figure'),
        [dash.dependencies.Input(component_id='drop41', component_property='value')]
)

def terror_over_time_reasons_attacktype(cr):

    df1 = data_clean[['iyear', cr,'attacktype1_txt']].groupby(['iyear', 'attacktype1_txt'], as_index=False).sum()
    # df2 = df1[df1[cr] == 1]

    fig = px.bar(df1, x="iyear", y = cr, color = 'attacktype1_txt' , title='Number Of Terrorist Activities WRT Reasons of Terror Each Year')

    return fig

@app.callback(
        dash.dependencies.Output(component_id='my_out_42', component_property='figure'),
        [dash.dependencies.Input(component_id='drop42', component_property='value')]
)

def terror_over_time_reasons_weapontype(cr):

    df1 = data_clean[['iyear', cr,'weaptype1_txt']].groupby(['iyear', 'weaptype1_txt'], as_index=False).sum()
    # df2 = df1[df1[cr] == 1]

    fig = px.bar(df1, x="iyear", y = cr, color = 'weaptype1_txt' , title='Number Of Terrorist Activities WRT Reasons of Terror Each Year')

    return fig


app.run_server(
    # debug = True,
    port = 8101,
    host = '0.0.0'
)