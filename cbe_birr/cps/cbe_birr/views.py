import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering without a display
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from django.shortcuts import render
from joblib import load
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import scipy.stats as stats
import seaborn as sns
from io import StringIO
from sqlalchemy import create_engine
import plotly.express as px
import plotly.io as pio
import base64
import calendar
import os
# Load the model for predictions
model = load('../cps_model/multioutput_model_pipeline.pkl')
cps_data = pd.read_csv('../cps_model/dcps.csv')
cust_data = pd.read_csv('../cps_model/cps_merged_cust.csv')

# # Create an engine using SQLAlchemy
# engine = create_engine('mysql+pymysql://root:root@localhost/cps')
# query = 'SELECT * FROM cps_merged'
# query1 = 'SELECT * FROM  cps_dcps'
# query2 = 'SELECT * FROM  cps_engaged_cust'
# cps_count = pd.read_sql(query2, engine)
# df_cps = pd.read_sql(query1,engine)
# data = pd.read_sql(query, engine)

data = pd.read_csv('../cps_model/cps_data.csv')
df_cps = pd.read_csv('../cps_model/cps_dcps.csv')
cps_count = pd.read_csv('../cps_model/cps_engaged_cust_count.csv')


data['BDA'] = pd.to_datetime(data['BDA'])
data.set_index('BDA', inplace=True)
data = data.drop_duplicates()
data = data.sort_index()
# Feature Engineering
data['day'] = data.index.day
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data['quarter'] = data.index.quarter
data['year'] = data.index.year

for target in ['NO_CUSTOMER', 'AMOUNT', 'NO_TXN']:
    decomposition = seasonal_decompose(data[target], model='additive', period=30)
    data[f'{target}_trend'] = decomposition.trend
    data[f'{target}_seasonal'] = decomposition.seasonal
    data[f'{target}_residual'] = decomposition.resid

data.dropna(inplace=True)

numerical_features = [
    'day', 'day_of_week', 'month', 'quarter', 'year',
    'NO_CUSTOMER_trend', 'NO_CUSTOMER_seasonal', 'NO_CUSTOMER_residual',
    'AMOUNT_trend', 'AMOUNT_seasonal', 'AMOUNT_residual',
    'NO_TXN_trend', 'NO_TXN_seasonal', 'NO_TXN_residual'
]
categorical_features = ['BRANCHNAME']
colors = ['Red', 'Brown', 'Gray', 'Black', 'Gold', 'Orange', 'Purple', 'Teal']


def generate_pie_chart(data, labels, x_labels, y_labels):
    my_title = None
    if y_labels == 'AMOUNT':
        my_title = 'Transaction Value'
    elif y_labels == 'NO_TXN':
        my_title = 'Transaction Volume'
    else:
        my_title = 'Customer Engagment'
    fig = px.pie(values=data, names=labels, 
                 title=my_title, 
                 color_discrete_sequence=['Teal', '#0f3460', '#16213e', '#e94560', '#1a1a2e', '#f5a5f5','Orange', '#ff2e63'])
    fig.update_traces(
        hovertemplate=f'{x_labels}: %{{label}}<br>{y_labels}: %{{value}}<br>Percentage: %{{percent:.2%}}<extra></extra>',
        textposition='inside', 
        textinfo='percent+label',
        showlegend=True,
        marker=dict(line=dict(color='black', width=1)))
    fig.update_layout(
        title_font_size=20, 
        margin=dict(t=60, b=60, l=20, r=60), 
        legend_title='Categories',  
        legend=dict(
            orientation="h",  
            yanchor="bottom",  
            y=-0.2,  
            xanchor="center",
            x=0.5  
        )
    )
    val_pie_chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    return val_pie_chart_html
def generate_bar_chart(data, labels, x_labels, y_labels):
    colors_yearly = ['Teal', '#0f3460', '#16213e', '#e94560', '#1a1a2e', '#f5a5f5', 'Orange', '#ff2e63']
    colors_monthly = ['Teal', '#0f3460', '#16213e', '#e94560', '#1a1a2e', '#f5a5f5', 'Orange', '#ff2e63', '#4caf50', '#ff9800', '#9c27b0', '#03a9f4']

    colors = []
    if x_labels == 'Month':
        labels = [calendar.month_abbr[int(month)] for month in labels]
        colors = colors_monthly
    else:
        colors = colors_yearly
    
    df = pd.DataFrame({
        x_labels: labels,
        y_labels: data
    })
    
    
    fig = px.bar(
        df, 
        x=x_labels, 
        y=y_labels, 
        color=x_labels, 
        color_discrete_sequence = colors,
        hover_data={x_labels: True, y_labels: True}  # Enable hover data
    )
    
    fig.update_traces(hovertemplate=f'{x_labels} = %{{x}}<br>{y_labels} = %{{y}}<extra></extra>')  # Customize hover info


    # Update the chart layout
    fig.update_traces(marker=dict(line=dict(color='black', width=1)))  # Add edge color to bars
     
    # Adjust the layout
    fig.update_layout(
        title_font_size=20,  # Set title font size
        margin=dict(t=60, b=60, l=60, r=60),  # Adjust margins for better spacing
        xaxis_title=x_labels,  # Set x-axis title
        yaxis_title='Customer Engagement',  # Set y-axis title
        legend_title=x_labels,  # Set the legend title
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",  # Anchor legend at the bottom
            y=-0.3,  # Position slightly below the chart
            xanchor="center",
            x=0.5  # Center the legend horizontally
        )
    )
    
    # Convert the Plotly figure to an HTML string
    bar_chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    return bar_chart_html


def generate_line_chart(df, df1):
    df.set_index('OPEN_DATE', inplace=True)
    df1.set_index('BUSINESSDATE', inplace=True)
    monthly_customers = df1['NO_CUSTOMER'].resample('ME').sum().reset_index()
    monthly_customers_held_trx = df['NO_CUSTOMER'].resample('ME').sum().reset_index()
    monthly_customers.columns = ['Date', 'Recruited']
    monthly_customers_held_trx.columns = ['Date', 'Held Transaction']
    combined_df = pd.merge(monthly_customers, monthly_customers_held_trx, on='Date')
    combined_df_melted = combined_df.melt(id_vars='Date', 
                                          value_vars=['Recruited', 'Held Transaction'], 
                                          var_name='Customer Type', 
                                          value_name='Number of Customers')
    fig = px.line(
        combined_df_melted, 
        x='Date', 
        y='Number of Customers', 
        color='Customer Type', 
        labels={'Date': 'Time'}, 
        color_discrete_sequence=['Black', 'Orange']
    )
    fig.update_traces(hovertemplate='%{x} <br>%{y} Customers<extra></extra>')
    fig.update_layout(
        title_font_size=20, 
        margin=dict(t=60, b=60, l=60, r=60), 
        xaxis_title='Time', 
        yaxis_title='Number of Customers', 
        legend_title='Customer Type',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    line_chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    return line_chart_html

def generate_moving_avg_chart(df):
    df.set_index('OPEN_DATE', inplace=True)
    df['monthly_avg'] = df['NO_CUSTOMER'].rolling(window=30).mean()
    df.reset_index(inplace=True)
    fig = px.line(
        df, 
        x='OPEN_DATE', 
        y='monthly_avg', 
        color='CBE_REGION', 
        title='Monthly Average of Customer Growth by Region',
        labels={'monthly_avg': 'Number of Customers', 'OPEN_DATE': 'Time'}, 
        color_discrete_sequence=['Purple', 'Gold', 'Black']
    )
    fig.update_traces(hovertemplate='Time: %{x}<br>Number of Customers: %{y}<br>Region: %{legendgroup}<extra></extra>')
    fig.update_layout(
        title_font_size=20,
        xaxis_title='Time',
        yaxis_title='Number of Customers',
        margin=dict(t=60, b=60, l=60, r=60),
        legend_title='CBE Region',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(tickangle=45) 
    )
    fig.update_xaxes(tickfont=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=15))
    moving_avg_chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    return moving_avg_chart_html



def generate_yearly_customer_growth_chart(df):
    region_yearly_customers = df.groupby('CBE_REGION')['NO_CUSTOMER'].resample('YE').sum().unstack(0)
    region_yearly_customers.reset_index(inplace=True)
    region_yearly_customers_melted = region_yearly_customers.melt(id_vars='OPEN_DATE', var_name='CBE_REGION', value_name='NO_CUSTOMER')
    fig = px.line(
        region_yearly_customers_melted, 
        x='OPEN_DATE', 
        y='NO_CUSTOMER', 
        color='CBE_REGION', 
        labels={'NO_CUSTOMER': 'Number of Customers', 'OPEN_DATE': 'Year'},
        color_discrete_sequence=['Purple', 'Gold', 'Black'],
        hover_name='CBE_REGION'  
    )
    fig.update_traces(hovertemplate='Year: %{x}<br>Number of Customers: %{y}<extra></extra>')
    fig.update_layout(
        title_font_size=20,
        xaxis_title='Year',
        yaxis_title='Number of Customers',
        margin=dict(t=60, b=60, l=60, r=60),  
        legend_title='Region',
        legend=dict(
            orientation="h",  
            yanchor="bottom",  
            y=-0.3,  
            xanchor="center",
            x=0.5  
        )
    )
    fig.update_xaxes(tickfont=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=15))
    yearly_customer_growth_chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    return yearly_customer_growth_chart_html


def cps_dashboard(request):
    
    acc_statues = cps_count.groupby('STATUS')['ENGAED_CUST'].sum()
    inact_acc = cps_count['ENGAED_CUST'].sum() - acc_statues['Active']
    count_cust = round((cps_count['ENGAED_CUST'].sum()/1000000), 1)
    acc_statues = round((acc_statues['Active']/1000000), 1)
    inact_acc = round((inact_acc/1000000), 1)
    
    recruiter = cps_count.groupby('RECRUITED_BY')['ENGAED_CUST'].sum()
    branch_cust = round((recruiter['Branch']/1000000), 1)

    branch_active_custs = cps_count.groupby(['STATUS','RECRUITED_BY'])['ENGAED_CUST'].sum()
    branch_active_cust = round((branch_active_custs.Active.Branch/1000000), 1)
    
    other_cust = round(((cps_count['ENGAED_CUST'].sum() - recruiter['Branch'])/1000000), 1)   
    
    
    df_cps['BUSINESSDATE'] = pd.to_datetime(df_cps['BUSINESSDATE'])
    df_cps['Year'] = df_cps['BUSINESSDATE'].dt.year
    yearly_data = df_cps.groupby('Year')['NO_CUSTOMER'].sum().reset_index()
    yearly_data_trx = df_cps.groupby('Year')['NO_TXN'].sum().reset_index()
    yearly_data_amt = df_cps.groupby('Year')['AMOUNT'].sum().reset_index()
    yearly_data_amt.sort_values(by = 'Year')
    yearly_data.sort_values(by = 'Year')
    yearly_data_trx.sort_values(by = 'Year')
    yearly_data_all = df_cps.groupby('Year').agg({'NO_CUSTOMER':'sum', 'NO_TXN':'sum', 'AMOUNT':'sum'}).reset_index()
    sums = yearly_data_all.sum()
    total_engage_cust = round((sums['NO_CUSTOMER'] /1000000), 1)
    total_trx_value = round((sums['AMOUNT']/1000000000000), 1)
    total_trx_volume = round((sums['NO_TXN']/1000000), 1)  
    

    val_pie_chart_html = generate_pie_chart(yearly_data_amt['AMOUNT'], yearly_data_amt['Year'], 'Year', 'AMOUNT')
    pie_chart_html = generate_pie_chart(yearly_data['NO_CUSTOMER'], yearly_data['Year'], 'Year', 'NO_CUSTOMER')
    vol_pie_chart_html = generate_pie_chart(yearly_data_trx['NO_TXN'], yearly_data_trx['Year'], 'Year', 'NO_TXN')
    
    
    val_bar_chart_html = generate_bar_chart(yearly_data_amt['AMOUNT'], yearly_data_amt['Year'], 'Year', 'AMOUNT')
    vol_bar_chart_html = generate_bar_chart(yearly_data_trx['NO_TXN'], yearly_data_trx['Year'], 'Year', 'NO_TXN')
    eng_bar_chart_html = generate_bar_chart(yearly_data['NO_CUSTOMER'], yearly_data['Year'], 'Year', 'NO_CUSTOMER')
    
    return render(request, 'dashboard.html', {
        'count_cust': count_cust,
        'acc_statues': acc_statues,
        'inact_acc': inact_acc,
        'branch_cust': branch_cust,
        'other_cust':other_cust,
        'total_trx_value':total_trx_value,
        'total_trx_volume':total_trx_volume,
        'branch_active_cust':branch_active_cust,
        'total_engage_cust':total_engage_cust,
        'pie_chart_html': pie_chart_html,
        'val_pie_chart_html': val_pie_chart_html,
        'vol_pie_chart_html':vol_pie_chart_html,
        'val_bar_chart_html':val_bar_chart_html,
        'vol_bar_chart_html':vol_bar_chart_html,
        'eng_bar_chart_html':eng_bar_chart_html,
    })

def feature_analysis(request):
    df_cps['BUSINESSDATE'] = pd.to_datetime(df_cps['BUSINESSDATE'])
    # Aggregate data by year
    df_cps['Year'] = df_cps['BUSINESSDATE'].dt.year
    yearly_data = df_cps.groupby('Year')['NO_CUSTOMER'].sum().reset_index()
    yearly_data_trx = df_cps.groupby('Year')['NO_TXN'].sum().reset_index()
    yearly_data_amt = df_cps.groupby('Year')['AMOUNT'].sum().reset_index()
    yearly_data_amt.sort_values(by = 'Year')
    yearly_data.sort_values(by = 'Year')
    yearly_data_trx.sort_values(by = 'Year')
    
    df_cps['Month'] = df_cps['BUSINESSDATE'].dt.month
    monthly_data_cust = df_cps.groupby('Month')['NO_CUSTOMER'].sum().reset_index()
    monthly_data_trx = df_cps.groupby('Month')['NO_TXN'].sum().reset_index()
    monthly_data_amt = df_cps.groupby('Month')['AMOUNT'].sum().reset_index()
    
    val_bar_chart_html = generate_bar_chart(yearly_data_amt['AMOUNT'], yearly_data_amt['Year'], 'Year', 'AMOUNT')
    vol_bar_chart_html = generate_bar_chart(yearly_data_trx['NO_TXN'], yearly_data_trx['Year'], 'Year', 'NO_TXN')
    eng_bar_chart_html = generate_bar_chart(yearly_data['NO_CUSTOMER'], yearly_data['Year'], 'Year', 'NO_CUSTOMER')
    
    val_bar_chart_month = generate_bar_chart(monthly_data_amt['AMOUNT'], monthly_data_amt['Month'], 'Month', 'AMOUNT')
    vol_bar_chart_month = generate_bar_chart(monthly_data_trx['NO_TXN'], monthly_data_trx['Month'], 'Month', 'NO_TXN')
    eng_bar_chart_month = generate_bar_chart(monthly_data_cust['NO_CUSTOMER'], monthly_data_cust['Month'], 'Month', 'NO_CUSTOMER')
    
    
    df1 = pd.read_csv('../cps_model/dcps.csv')
    df = pd.read_csv('../cps_model/cps_merged_cust.csv')
    df1['BUSINESSDATE'] = pd.to_datetime(df1['BUSINESSDATE'])
    df['OPEN_DATE'] = pd.to_datetime(df['OPEN_DATE'])
    line_chart = generate_line_chart(df, df1)
    
      
    # cust_growth_reg = generate_moving_avg_chart(df)
    cust_growth_reg = generate_yearly_customer_growth_chart(df)
    
    return render(request, "distribution.html",{
        'val_bar_chart_html': val_bar_chart_html,
        'vol_bar_chart_html': vol_bar_chart_html,
        'eng_bar_chart_html': eng_bar_chart_html,
        
        'val_bar_chart_month': val_bar_chart_month,
        'vol_bar_chart_month': vol_bar_chart_month,
        'eng_bar_chart_month': eng_bar_chart_month,
        
        'line_chart':line_chart,
        'cust_growth_reg':cust_growth_reg,
    })







def predictor(request):    
    df = cps_data['BRANCHNAME'].unique()
    branches = df.tolist()  # Convert column to a list
    return render(request, "main.html", {'branches': branches})

def formInfo(request):
    data.reset_index(inplace=True)
    data.set_index(['BDA', 'BRANCHNAME'], inplace=True)
    branch_name = request.GET['branch_name']
    date1 = request.GET['date1']
    date2 = request.GET['date2']
    district_name = cps_data.loc[cps_data['BRANCHNAME'] == branch_name, 'DISTRICTNAME'].values[0]
    date_range = pd.date_range(start=date1, end=date2)
    date_range = [date1, date2]
    result = None
    branch_data = data
    results = []
    for date1 in date_range:
        future_data = pd.DataFrame({'BDA': [pd.to_datetime(date1)], 'BRANCHNAME': [branch_name]})
       
        future_data.set_index('BDA', inplace=True)
        future_data['day'] = future_data.index.day
        
        future_data['day_of_week'] = future_data.index.dayofweek
        future_data['month'] = future_data.index.month
        future_data['quarter'] = future_data.index.quarter
        future_data['year'] = future_data.index.year
        
        numerical_featuress = []
        categorical_featuress = []
        
        future_data.set_index('BRANCHNAME', append=True, inplace=True)  # Set 'BRANCHNAME' as secondary index
        for target in ['NO_CUSTOMER', 'AMOUNT', 'NO_TXN']:            
                future_data[f'{target}_trend'] = data[f'{target}_trend']
                future_data[f'{target}_seasonal'] = data[f'{target}_seasonal']
                future_data[f'{target}_residual'] = data[f'{target}_residual']


                categorical_featuress = ['BRANCHNAME']
                numerical_featuress = ['day', 'day_of_week', 'month', 'quarter', 'year',
                                     f'{target}_trend', f'{target}_seasonal', f'{target}_residual']
                
                
                # Trend data for the specific branch
                trend_data = branch_data[[f'{target}_trend']].dropna().reset_index()


                trend_data['BDA_ordinal'] = trend_data['BDA'].map(pd.Timestamp.toordinal)

                X_trend = trend_data[['BDA_ordinal']]
                y_trend = trend_data[f'{target}_trend']

                trend_model = LinearRegression()
                trend_model.fit(X_trend, y_trend)


                future_date_ordinal = pd.to_datetime(date1).toordinal()
                future_trend = trend_model.predict([[future_date_ordinal]])

#                 Seasonality 
                seasonal_data = branch_data[[f'{target}_seasonal']].dropna().reset_index()
                seasonal_data['BDA_ordinal'] = seasonal_data['BDA'].map(pd.Timestamp.toordinal)

                X_seasonal = seasonal_data[['BDA_ordinal']]
                y_seasonal = seasonal_data[f'{target}_seasonal']

                seasonal_model = LinearRegression()
                seasonal_model.fit(X_seasonal, y_seasonal)
                future_seasonal = seasonal_model.predict([[future_date_ordinal]])
                
                
#                 Residual
                residual_data = branch_data[[f'{target}_residual']].dropna().reset_index()
                residual_data['BDA_ordinal'] = residual_data['BDA'].map(pd.Timestamp.toordinal)

                X_residual = residual_data[['BDA_ordinal']]
                y_residual = residual_data[f'{target}_residual']

                residual_model = LinearRegression()
                residual_model.fit(X_residual, y_residual)
                future_residual = residual_model.predict([[future_date_ordinal]])
                
                

                historical_residuals = 0
                future_data[f'{target}_trend'] = future_trend
                future_data[f'{target}_seasonal'] = future_seasonal
                historical_residuals = data[data.index.get_level_values('BRANCHNAME') == branch_name][f'{target}_residual'].max()
                future_data[f'{target}_residual'] = historical_residuals
                
                if (data.index.get_level_values('BRANCHNAME') == branch_name).any() and (data.index.get_level_values('BDA') == pd.to_datetime(date1)).any():
                    future_data[f'{target}_trend'] = data.loc[(data.index.get_level_values('BRANCHNAME') == branch_name) & (data.index.get_level_values('BDA') == pd.to_datetime(date1)), f'{target}_trend'].values
                    future_data[f'{target}_seasonal'] = data.loc[(data.index.get_level_values('BRANCHNAME') == branch_name) & (data.index.get_level_values('BDA') == pd.to_datetime(date1)), f'{target}_seasonal'].values
                    future_data[f'{target}_residual'] = data.loc[(data.index.get_level_values('BRANCHNAME') == branch_name) & (data.index.get_level_values('BDA') == pd.to_datetime(date1)), f'{target}_residual'].values

                else:  
                    future_data = future_data

        future_data.dropna(inplace=True)

        # Encode the branch name using OneHotEncoder
        ohe = model.named_steps['preprocessor'].named_transformers_['cat']
        encoded_branch_name = ohe.transform([[branch_name]]).toarray()
        encoded_branch_df = pd.DataFrame(encoded_branch_name, columns=ohe.get_feature_names_out(['BRANCHNAME']))
        
        

        
        df_reset = future_data.reset_index()
        df_reset.set_index('BDA', inplace=True)
        numerical_data = df_reset[numerical_featuress]
        categorical_data = df_reset[categorical_features]
        future_data_processed = pd.concat([numerical_data, categorical_data], axis=1)
        future_data_processed = df_reset


        # Make multi-output predictions
        result = model.predict(future_data_processed)
        results.append([
            date1,           # 'date'
            result[0, 0],    # 'NO_CUSTOMER'
            result[0, 2],    # 'NO_TXN'
            result[0, 1]     # 'AMOUNT'
        ])
        
        def format_scientific(value):
            return "{:.2e}".format(float(value))
        formatted_result = [[row[0], format_scientific(row[1]), format_scientific(row[2]), format_scientific(row[3])] for row in results]

        formatted_result = [[row[0], round(float(row[1]),2), round(float(row[2]), 2), round(float(row[3]), 2)] for row in results]
        
        
        future_data.reset_index(level='BRANCHNAME', inplace=True)
    data.reset_index(level='BRANCHNAME', inplace=True)
    return render(request, 'result.html', {
        'branch_name': branch_name,
        'district_name': district_name,
        'results': formatted_result
    })
    
    
    
def batch_predictor(request):
    results = []
    data.reset_index(inplace=True)
    data.set_index(['BDA', 'BRANCHNAME'], inplace=True)
    if request.method == 'POST':
        file = request.FILES['file']
        df = pd.read_csv(file)
        df = df[['BDA', 'BRANCHNAME']]
        df['BDA'] = pd.to_datetime(df['BDA'])

        # Sort the DataFrame by 'date' in ascending order
        df = df.sort_values(by='BDA', ascending=True)
        
        df.reset_index(drop=True, inplace=True)
        
        # first_date = df['BDA'].min()
        # last_date = df['BDA'].max()
        # date_range = [first_date, last_date]
        
        result = None
        branch_data = data
        results = []
        for i , row in df.iterrows():
            date1 = row['BDA']  # Access the date value
            branch_name = row['BRANCHNAME'] 
            future_data = pd.DataFrame({'BDA': [pd.to_datetime(date1)], 'BRANCHNAME': [branch_name]})
       
            future_data.set_index('BDA', inplace=True)
            future_data['day'] = future_data.index.day
            
            future_data['day_of_week'] = future_data.index.dayofweek
            future_data['month'] = future_data.index.month
            future_data['quarter'] = future_data.index.quarter
            future_data['year'] = future_data.index.year
            
            numerical_featuress = []
            categorical_featuress = []
            future_data.set_index('BRANCHNAME', append=True, inplace=True)  # Set 'BRANCHNAME' as secondary index
            for target in ['NO_CUSTOMER', 'AMOUNT', 'NO_TXN']:            
                    future_data[f'{target}_trend'] = data[f'{target}_trend']
                    future_data[f'{target}_seasonal'] = data[f'{target}_seasonal']
                    future_data[f'{target}_residual'] = data[f'{target}_residual']


                    categorical_featuress = ['BRANCHNAME']
                    numerical_featuress = ['day', 'day_of_week', 'month', 'quarter', 'year',
                                        f'{target}_trend', f'{target}_seasonal', f'{target}_residual']
                    
                    
                    # Trend data for the specific branch
                    trend_data = branch_data[[f'{target}_trend']].dropna().reset_index()


                    trend_data['BDA_ordinal'] = trend_data['BDA'].map(pd.Timestamp.toordinal)

                    X_trend = trend_data[['BDA_ordinal']]
                    y_trend = trend_data[f'{target}_trend']

                    trend_model = LinearRegression()
                    trend_model.fit(X_trend, y_trend)


                    future_date_ordinal = pd.to_datetime(date1).toordinal()
                    future_trend = trend_model.predict([[future_date_ordinal]])

    #                 Seasonality 
                    seasonal_data = branch_data[[f'{target}_seasonal']].dropna().reset_index()
                    seasonal_data['BDA_ordinal'] = seasonal_data['BDA'].map(pd.Timestamp.toordinal)

                    X_seasonal = seasonal_data[['BDA_ordinal']]
                    y_seasonal = seasonal_data[f'{target}_seasonal']

                    seasonal_model = LinearRegression()
                    seasonal_model.fit(X_seasonal, y_seasonal)
                    future_seasonal = seasonal_model.predict([[future_date_ordinal]])
                    
                    
    #                 Residual
                    residual_data = branch_data[[f'{target}_residual']].dropna().reset_index()
                    residual_data['BDA_ordinal'] = residual_data['BDA'].map(pd.Timestamp.toordinal)

                    X_residual = residual_data[['BDA_ordinal']]
                    y_residual = residual_data[f'{target}_residual']

                    residual_model = LinearRegression()
                    residual_model.fit(X_residual, y_residual)
                    future_residual = residual_model.predict([[future_date_ordinal]])
                    
                    

                    historical_residuals = 0
                    future_data[f'{target}_trend'] = future_trend
                    future_data[f'{target}_seasonal'] = future_seasonal
                    historical_residuals = data[data.index.get_level_values('BRANCHNAME') == branch_name][f'{target}_residual'].max()
                    future_data[f'{target}_residual'] = historical_residuals
                    
                    if (data.index.get_level_values('BRANCHNAME') == branch_name).any() and (data.index.get_level_values('BDA') == pd.to_datetime(date1)).any():
                        future_data[f'{target}_trend'] = data.loc[(data.index.get_level_values('BRANCHNAME') == branch_name) & (data.index.get_level_values('BDA') == pd.to_datetime(date1)), f'{target}_trend'].values
                        future_data[f'{target}_seasonal'] = data.loc[(data.index.get_level_values('BRANCHNAME') == branch_name) & (data.index.get_level_values('BDA') == pd.to_datetime(date1)), f'{target}_seasonal'].values
                        future_data[f'{target}_residual'] = data.loc[(data.index.get_level_values('BRANCHNAME') == branch_name) & (data.index.get_level_values('BDA') == pd.to_datetime(date1)), f'{target}_residual'].values

                    else:  
                        future_data = future_data

            future_data.dropna(inplace=True)
            
            # Encode the branch name using OneHotEncoder
            ohe = model.named_steps['preprocessor'].named_transformers_['cat']
            encoded_branch_name = ohe.transform([[branch_name]]).toarray()
            encoded_branch_df = pd.DataFrame(encoded_branch_name, columns=ohe.get_feature_names_out(['BRANCHNAME']))

            df_reset = future_data.reset_index()
            df_reset.set_index('BDA', inplace=True)
            numerical_data = df_reset[numerical_featuress]
            categorical_data = df_reset[categorical_features]
            future_data_processed = pd.concat([numerical_data, categorical_data], axis=1)
            future_data_processed = df_reset


            # Make multi-output predictions
            result = model.predict(future_data_processed)
            results.append([
                date1,           # 'date'
                
                branch_name,
                round(float(result[0, 0]), 2),    # 'NO_CUSTOMER'
                round(float(result[0, 2]), 2),    # 'NO_TXN'
                round(float(result[0, 1]), 2)     # 'AMOUNT'
            ])
            future_data.reset_index(level='BRANCHNAME', inplace=True)
            
    data.reset_index(inplace=True)  # This will reset all indices
    return render(request, 'batch_predict.html', {'results': results})


def data_exploration(request):
    df = df_cps
    
    # Select top 5 records
    top_5 = df.head()
    bot_5 = df.tail()

    # Get basic information about the data (capturing info output)
    buffer = StringIO()
    df.info(buf=buffer)
    data_info = buffer.getvalue()

    # Get summary statistics
    summary_stats = df.describe().to_dict()
    # Rename problematic keys (like '25%', '50%', '75%')
    for feature, stats in summary_stats.items():
        stats['Q1'] = stats.pop('25%')
        stats['Midean'] = stats.pop('50%')
        stats['Q3'] = stats.pop('75%')
    
    
    # Convert top 5 data to dictionary
    results = top_5.to_dict(orient='records')
    bot_5 = bot_5.to_dict(orient='records')

    # Pass all data to the template
    return render(request, 'exploration.html', {
        'results': results,
        'bot_5': bot_5,
        'data_info': data_info,
        'summary_stats': summary_stats,
    })