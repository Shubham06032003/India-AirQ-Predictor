import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

@st.cache_data
def load_clean_data():
    data = pd.read_csv('data/city_day.csv')
    data.fillna(data.median(numeric_only=True), inplace=True)
    data.dropna(subset=['AQI'], inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])

    return data

@st.cache_resource
def load_models():
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    le_city = pickle.load(open('model/city_encoder.pkl', 'rb'))
    feature_names = pickle.load(open('model/feature_names.pkl', 'rb'))

    return model, scaler, le_city, feature_names


@st.cache_data
def get_pollutant_stats(data):
    """Calculate min, max, mean for each pollutant and year"""
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                  'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

    stats = {}

    # Pollutant stats
    for col in pollutants:
        if col in data.columns:
            stats[col] = {
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'mean': float(data[col].mean())
            }

    # ✅ FIX: Extract year from Date column (which is datetime)
    try:
        # If Date is datetime, extract year
        if pd.api.types.is_datetime64_any_dtype(data['Date']):
            years = data['Date'].dt.year
        else:
            # If Date is already integer year
            years = data['Date']

        stats['Year'] = {
            'min': int(years.min()),
            'max': int(years.max()),
            'mean': int(years.mean())
        }
    except KeyError:
        # Fallback if no Date column
        st.warning("No Date column found. Using default year range.")
        stats['Year'] = {
            'min': 2015,
            'max': 2020,
            'mean': 2018
        }

    return stats

def create_model(model,scaler,le_city,feature_names,data):
    st.markdown("""
        <style>
        /* Make sure sidebar content can scroll */
        section[data-testid="stSidebar"] > div {
            overflow-y: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar for inputs
    st.sidebar.header("🎯 Prediction Inputs")

    # City selection
    city = st.sidebar.selectbox(
        "Select City",
        options=le_city.classes_,
        help="Choose the city for AQI prediction"
    )

    st.sidebar.markdown("### Pollutant Levels")

    predict_btn = st.sidebar.button("🔮 Predict AQI", type="primary", width='stretch')


    st.sidebar.markdown("---")  # Divider

    # ✅ Get stats for dynamic ranges
    stats = get_pollutant_stats(data)

    st.sidebar.markdown("### 📅 Time Period")
    year = st.sidebar.slider(
        "Year",
        min_value=stats['Year']['min'],
        max_value=stats['Year']['max'],
        value=stats['Year']['mean'],
        step=1
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Adjust Pollutant Parameters Below")


    # Pollutant inputs
    pm25 = st.sidebar.slider(
        "PM2.5 (µg/m³)",
        min_value=stats['PM2.5']['min'],
        max_value=stats['PM2.5']['max'],
        value=stats['PM2.5']['mean'],
        step=1.0
    )

    pm10 = st.sidebar.slider(
        "PM10 (µg/m³)",
        min_value=stats['PM10']['min'],
        max_value=stats['PM10']['max'],
        value=stats['PM10']['mean'],
        step=1.0
    )

    no = st.sidebar.slider(
        "NO (µg/m³)",
        min_value=stats['NO']['min'],
        max_value=stats['NO']['max'],
        value=stats['NO']['mean'],
        step=1.0
    )

    no2 = st.sidebar.slider(
        "NO2 (µg/m³)",
        min_value=stats['NO2']['min'],
        max_value=stats['NO2']['max'],
        value=stats['NO2']['mean'],
        step=1.0
    )

    nox = st.sidebar.slider(
        "NOx (ppb)",
        min_value=stats['NOx']['min'],
        max_value=stats['NOx']['max'],
        value=stats['NOx']['mean'],
        step=1.0
    )

    nh3 = st.sidebar.slider(
        "NH3 (µg/m³)",
        min_value=stats['NH3']['min'],
        max_value=stats['NH3']['max'],
        value=stats['NH3']['mean'],
        step=1.0
    )

    co = st.sidebar.slider(
        "CO (mg/m³)",
        min_value=stats['CO']['min'],
        max_value=stats['CO']['max'],
        value=stats['CO']['mean'],
        step=0.1
    )

    so2 = st.sidebar.slider(
        "SO2 (µg/m³)",
        min_value=stats['SO2']['min'],
        max_value=stats['SO2']['max'],
        value=stats['SO2']['mean'],
        step=1.0
    )

    o3 = st.sidebar.slider(
        "O3 (µg/m³)",
        min_value=stats['O3']['min'],
        max_value=stats['O3']['max'],
        value=stats['O3']['mean'],
        step=1.0
    )

    benzene = st.sidebar.slider(
        "Benzene (µg/m³)",
        min_value=stats['Benzene']['min'],
        max_value=stats['Benzene']['max'],
        value=stats['Benzene']['mean'],
        step=0.1
    )

    toluene = st.sidebar.slider(
        "Toluene (µg/m³)",
        min_value=stats['Toluene']['min'],
        max_value=stats['Toluene']['max'],
        value=stats['Toluene']['mean'],
        step=0.1
    )

    xylene = st.sidebar.slider(
        "Xylene (µg/m³)",
        min_value=stats['Xylene']['min'],
        max_value=stats['Xylene']['max'],
        value=stats['Xylene']['mean'],
        step=0.1
    )

    # Main area - show prediction
    if predict_btn:
        with st.spinner("Predicting..."):
            try:
                # Encode city
                city_encoded = le_city.transform([city])[0]

                # Prepare input (match training feature order!)
                input_data = pd.DataFrame({
                    'City_encoded': [city_encoded],
                    'Date': [year],
                    'PM2.5': [pm25],
                    'PM10': [pm10],
                    'NO': [no],
                    'NO2': [no2],
                    'NOx': [nox],
                    'NH3': [nh3],
                    'CO': [co],
                    'SO2': [so2],
                    'O3': [o3],
                    'Benzene': [benzene],
                    'Toluene': [toluene],
                    'Xylene': [xylene]
                })

                input_data = input_data[feature_names]

                # Scale features
                input_scaled = scaler.transform(input_data)

                # Predict
                predicted_aqi = model.predict(input_scaled)[0]

                # Display result
                st.markdown("## 🎯 Prediction Result")

                # Determine AQI category
                if predicted_aqi <= 50:
                    category = "Good"
                    color = "green"
                    emoji = "😊"
                elif predicted_aqi <= 100:
                    category = "Moderate"
                    color = "yellow"
                    emoji = "😐"
                elif predicted_aqi <= 200:
                    category = "Unhealthy"
                    color = "orange"
                    emoji = "😷"
                elif predicted_aqi <= 300:
                    category = "Very Unhealthy"
                    color = "red"
                    emoji = "🤢"
                else:
                    category = "Hazardous"
                    color = "purple"
                    emoji = "☠️"

                # Show in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted AQI", f"{predicted_aqi:.1f}")
                with col2:
                    st.metric("Category", category)
                with col3:
                    st.markdown(f"### {emoji}")

                # Color-coded alert
                st.markdown(f"""
                <div style="padding: 20px; 
                            background-color: {color}; 
                            border-radius: 10px; 
                            text-align: center;">
                    <h2 style="color: white;">Air Quality: {category}</h2>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # Add after prediction result
    if predict_btn:
        # (prediction code here...)

        st.markdown("---")
        st.markdown("## 📊 Visualizations")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_aqi,
            title={'text': "AQI Gauge"},
            gauge={
                'axis': {'range': [0, 500]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "yellow"},
                    {'range': [100, 200], 'color': "orange"},
                    {'range': [200, 300], 'color': "red"},
                    {'range': [300, 500], 'color': "purple"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, width='stretch')

        # Pollutant bar chart
        pollutants_data = {
            'Pollutant': ['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'SO2', 'O3'],
            'Level': [pm25, pm10, no, no2, co, so2, o3]
        }
        fig_bar = px.bar(pollutants_data, x='Pollutant', y='Level',
                         title="Current Pollutant Levels",
                         color='Level', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, width='stretch')


def historical_tab(data):
    """Historical data tab content"""

    st.markdown("## 📈 Historical AQI Trends")

    # Filter by city
    city_filter = st.selectbox("Select City for History", data['City'].unique())

    city_data = data[data['City'] == city_filter]

    # Line chart
    fig_line = px.line(city_data, x='Date', y='AQI',
                       title=f'AQI Trends in {city_filter}',
                       labels={'AQI': 'Air Quality Index', 'Date': 'Date'})
    fig_line.update_traces(line_color='#1f77b4', line_width=2)
    fig_line.update_layout(hovermode='x unified')
    st.plotly_chart(fig_line, width='stretch')

    # Show statistics
    st.markdown("### 📊 City Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average AQI", f"{city_data['AQI'].mean():.1f}")
    with col2:
        st.metric("Max AQI", f"{city_data['AQI'].max():.1f}")
    with col3:
        st.metric("Min AQI", f"{city_data['AQI'].min():.1f}")


def show_footer():
    """Beautiful gradient footer with stats and links"""
    st.markdown("---")

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 1.5rem;
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    ">
        <h2 style="color: white; text-align: center; margin-bottom: 0.5rem; font-weight: 600;">
            🌍 AQI Prediction Dashboard
        </h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center; font-size: 1.1rem; margin-bottom: 2rem;">
            Track. Predict. Protect. Your Air Quality Companion
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row with gradient background styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0 1.5rem 1.5rem 1.5rem;
        border-radius: 0 0 15px 15px;
        margin-top: -1rem;
    ">
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 12px; text-align: center;">
            <p style="font-size: 2.5rem; margin: 0; font-weight: bold; color: white;">90.9%</p>
            <p style="font-size: 1rem; margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.95);">Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 12px; text-align: center;">
            <p style="font-size: 2.5rem; margin: 0; font-weight: bold; color: white;">10+</p>
            <p style="font-size: 1rem; margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.95);">Cities Covered</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 12px; text-align: center;">
            <p style="font-size: 2.5rem; margin: 0; font-weight: bold; color: white;">29K+</p>
            <p style="font-size: 1rem; margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.95);">Data Records</p>
        </div>
        """, unsafe_allow_html=True)

    # Bottom section with links
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0 1rem 0;">
        <p style="font-size: 1rem; color: rgba(255,255,255,0.9); margin-bottom: 1rem;">
            Built with ❤️ using <b>Streamlit</b> & <b>Random Forest ML</b>
        </p>
        <div style="background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px; display: inline-block; margin-bottom: 1rem;">
            <a href="https://github.com/Shubham06032003/India-AirQ-Predictor" target="_blank" style="color: white; text-decoration: none; margin: 0 1rem; font-weight: 500;">
                🔗 GitHub
            </a>
            <span style="color: rgba(255,255,255,0.5);">|</span>
            <a href="https://www.linkedin.com/in/shubham-panwar-b1a360238/" target="_blank" style="color: white; text-decoration: none; margin: 0 1rem; font-weight: 500;">
                💼 LinkedIn
            </a>
            <span style="color: rgba(255,255,255,0.5);">|</span>
            <p style="color: white; text-decoration: none; margin: 0 1rem; font-weight: 500;">
                ✉️ Contact : Shub.p.2003@gmail.com
            </p>
        </div>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin: 0;">
            © 2025 AQI Dashboard | Made by <b>Shubham Panwar</b> | Open Source Project
        </p>
    </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title = 'AirQuality Predictor',
        page_icon = '🌍',
        layout = 'wide',
    )

    with st.container():
        st.title('🌍Air Quality Index Dashboard')
        st.subheader('''Monitor air quality in real-time across major Indian cities

Track. Predict. Protect.
Your personal air quality companion powered by AI''')

    data = load_clean_data()
    model, scaler, le_city, feature_names = load_models()

    tab1, tab2 = st.tabs(["🔮 Prediction", "📈 Historical Data"])
    with tab1:
        create_model(model,scaler,le_city,feature_names,data)

    with tab2:
        historical_tab(data)

    # Footer
    show_footer()





if __name__ == '__main__':

    main()
