import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import openpyxl
import plotly.graph_objs as go

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.markdown(
    "<h1 style='text-align: center; font-size: 32px;'>Welcome to this dashboard created to analyze the dataset of Getaround delays.</h1>",
    unsafe_allow_html=True
)

st.sidebar.header('Dashboard')
st.sidebar.title("Summary")

pages = ["Project Goals", "Dataset", "EDA","Threshold and Scope effect"]

page = st.sidebar.radio("Go to the page :", pages)
st.sidebar.markdown('''
---
Created by Seddik AMROUN.
''')
@st.cache_data
def load_data():
    df = pd.read_excel('get_around_delay_analysis.xlsx')
    return df 

#data_load_state = st.text('Loading data...')
data = load_data()
#data_load_state.text("Loading data...done!")

if page == pages[0] :
    st.write("### Project  Goals")
    st.image("GET.png")
    st.write("""When renting a car through Getaround, users must complete a check-in at the start and a checkout at the end of the rental to:
             Assess the car's condition and report any damages.
             Compare fuel levels.
             Measure kilometers driven.
             """)

    st.write("""Check-in and checkout can be done via:
             Mobile: Both driver and owner sign the agreement on the owner's smartphone.
             Connect: Driver opens the car with their smartphone, no meeting with the owner.
             """)
    
    st.write("""Drivers sometimes return cars late, causing issues for the next rental. To mitigate this, we plan to implement a minimum delay between rentals to avoid scheduling conflicts. 
    This solution aims to reduce customer dissatisfaction but might impact revenue.""")
    
    


elif page == pages[1]:
    st.write("### Dataset")
    
    st.dataframe(data.head(10))
    st.write("Shape of the dataset :")
    st.write(data.shape)
    if st.checkbox("Some statistics") :
        cols=["delay_at_checkout_in_minutes","previous_ended_rental_id","time_delta_with_previous_rental_in_minutes"] 
        st.write(data[cols].describe())
    if st.checkbox("The missing values") : 
        st.dataframe((data.isna().sum()/data.shape[0]*100).sort_values(ascending=False))
    
elif page == pages[2]:
    st.write("### Exploration Data Analysis")
    c1, c2 = st.columns((3,3))
    with c1:
        fig,ax= plt.subplots()
        data.checkin_type.value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Checkin Type Distribution')
        st.pyplot(fig)
    with c2:
        fig,ax= plt.subplots()
        data.state.value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('state distribution')
        st.pyplot(fig)
    
    c3, c4 = st.columns((2,2))
    with c3:
        data['status'] = data['delay_at_checkout_in_minutes'].apply(lambda x: 'late' if x > 0 else 'on_time')
        grouped = pd.DataFrame(data).groupby(['checkin_type', 'status']).size().unstack(fill_value=0).reset_index()
        melted = pd.melt(grouped, id_vars='checkin_type', var_name='status', value_name='count')
        plt.figure(figsize=(8, 6))
        sns.barplot(data=melted, x='checkin_type', y='count', hue='status')
        plt.title('Checkin Type by Late or On Time')
        plt.xlabel('Checkin Type')
        plt.ylabel('Count')
        plt.legend(title='Status')
        st.pyplot(plt)
    with c4:
        grouped = data.groupby(['state', 'checkin_type']).size().unstack(fill_value=0).reset_index()
        melted = grouped.melt(id_vars='state', var_name='checkin_type', value_name='count')
        plt.figure(figsize=(8, 6))
        sns.barplot(data=melted, x='state', y='count', hue='checkin_type')
        plt.title('State by checkin_type')
        plt.xlabel('State')
        plt.legend(title='checkin_type')
        st.pyplot(plt)        
else:
    st.write("### More Analysis")
    # Difference between delay at checkout and the delta with previous rental
    data['minutes_passed_checkin_time'] = data['delay_at_checkout_in_minutes'] - data['time_delta_with_previous_rental_in_minutes']
    impacted_df = data[data["time_delta_with_previous_rental_in_minutes"].notna()]

    st.header("How many impacted and solved rentals cases depending on threshold and scope ?")

    threshold_range = np.arange(0, 1440, step=30)
    impacted_list_mobile = []
    impacted_list_connect = []
    impacted_list_total = []
    solved_list_mobile = []
    solved_list_connect = []
    solved_list_total = []

    solved_list = []
    for t in range(1440):
        connect_impact = impacted_df[impacted_df['checkin_type'] == 'connect']
        mobile_impact = impacted_df[impacted_df['checkin_type'] == 'mobile']
        connect_impact = connect_impact[connect_impact['time_delta_with_previous_rental_in_minutes'] < t]
        mobile_impact = mobile_impact[mobile_impact['time_delta_with_previous_rental_in_minutes'] < t]
        impacted = impacted_df[impacted_df['time_delta_with_previous_rental_in_minutes'] < t]
        impacted_list_connect.append(len(connect_impact))
        impacted_list_mobile.append(len(mobile_impact))
        impacted_list_total.append(len(impacted))

        solved = impacted_df[data['minutes_passed_checkin_time'] > 0]
        connect_solved = solved[solved['checkin_type'] == 'connect']
        mobile_solved = solved[solved['checkin_type'] == 'mobile']
        connect_solved = connect_solved[connect_solved['delay_at_checkout_in_minutes'] < t]
        mobile_solved = mobile_solved[mobile_solved['delay_at_checkout_in_minutes'] < t]
        solved = solved[solved['delay_at_checkout_in_minutes'] < t]
        solved_list_connect.append(len(connect_solved))
        solved_list_mobile.append(len(mobile_solved))
        solved_list_total.append(len(solved))


    # Convert range to a list for 'x' argument
    x_values = list(range(1440))

    co1,co2 = st.columns(2)
    with co1:

        # Creation of the 3 traces
        total_impacted_cars = go.Scatter(x=x_values, y=impacted_list_total, name='All cars')
        impacted_connect_cars = go.Scatter(x=x_values, y=impacted_list_connect, name='Connect cars')
        impacted_mobile_cars = go.Scatter(x=x_values, y=impacted_list_mobile, name='Mobile cars')

        # Create layout for the plot
        layout = go.Layout(
            title='Number of impacted cases by threshold',
            xaxis=dict(title='Threshold in minutes'),
            yaxis=dict(title='Number of impacted cases'),
            xaxis_tickvals=list(range(0, 1440, 30)),
            legend=dict(orientation='h', yanchor='bottom', xanchor='right',y=1.02, x=1)
        )

        # Create figure and add traces to it
        fig = go.Figure(data=[total_impacted_cars, impacted_connect_cars, impacted_mobile_cars], layout=layout)
        st.plotly_chart(fig, width = 800, height = 600, use_container_width=True)

    
    with co2:

        # Creation of the 3 traces
        total_solved_cars = go.Scatter(x=x_values, y=solved_list_total, name='All cars')
        connect_solved_cars = go.Scatter(x=x_values, y=solved_list_connect, name='Connect cars')
        mobile_solved_cars = go.Scatter(x=x_values, y=solved_list_mobile, name='Mobile cars')

        # Create layout for the plot
        layout = go.Layout(
            title='Threshold solved cases number',
            xaxis=dict(title='Threshold in minutes'),
            yaxis=dict(title='Number of cases solved'),
            xaxis_tickvals=list(range(0, 1440, 30)),
            legend=dict(orientation='h', yanchor='bottom', xanchor='right',y=1.02, x=1)
        )

        # Create figure and add traces to it
        fig = go.Figure(data=[total_solved_cars, connect_solved_cars, mobile_solved_cars], layout=layout)
        st.plotly_chart(fig, width = 800, height = 600, use_container_width=True)
        #Sélection de l'option dans le menu déroulant
        

    st.subheader("Graph analysis")
    st.markdown("""* We see the curve of solved cases flatten out after **120-140 minutes**, even up to 180 minutes. * => a much higher threshold could solve many problem cases.
    * Notice that if we increase the treshold, it influences the number of cars available and so the benefits.  
    * Like everywhere, it's a matter of compromise between those two features : reality facts and benefits.   
    * Knowing this, :red[**140 minutes**] threshold seems to be a good compromise for business.""")
    st.write("")
    st.header("Threshold and Scope effects")
    st.markdown("Play yourself with the app here: adjust the threshold to see the effects on data")
        ## Threshold and scope form
    with st.form("threshold_testing"):
        threshold = st.slider("Choose threshold in minutes", 0,1440,0)
        checkin_type = st.radio("Choose desired checkin type", ["All", "Connect", "Mobile"])
        submit = st.form_submit_button("Check it out !")
        if submit:
            st.markdown(f"With a threshold of **{threshold}** and for **{checkin_type}** scope")
            if checkin_type == "All":
                st.metric(f"The number of cases impacted is :",impacted_list_total[threshold])
                st.metric("The number of cases solved is :",solved_list_total[threshold])
            elif checkin_type == "Connect":
                st.metric(f"The number of cases impacted is :",impacted_list_connect[threshold])
                st.metric("The number of cases solved is :",solved_list_connect[threshold])
            else :
                st.metric(f"The number of cases impacted is :",impacted_list_mobile[threshold])
                st.metric("The number of cases solved is :",solved_list_mobile[threshold])


        