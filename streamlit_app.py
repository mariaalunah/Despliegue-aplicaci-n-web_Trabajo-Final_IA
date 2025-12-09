
import streamlit as st
import pandas as pd
import joblib

# --- Load pre-trained artifacts ---
@st.cache_resource
def _load_model_artifacts_internal():
    model = joblib.load('best_bagging_model.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    return model, label_encoders, feature_columns

try:
    model, label_encoders, feature_columns = _load_model_artifacts_internal()
except FileNotFoundError:
    st.error("Error: Missing model artifact file. Please ensure 'best_bagging_model.joblib', 'label_encoders.joblib', and 'feature_columns.joblib' are in the current directory.")
    st.stop() # Stop the app if essential files are missing

# --- Preprocessing function for new data ---
def preprocess_input(input_df, label_encoders, feature_columns):
    df = input_df.copy()

    # Convert TotalCharges to numeric, handling potential errors from user input (if not already handled by st.number_input)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # If TotalCharges is still NaN (e.g., if user entered 0 for tenure and MonthlyCharges, resulting in NaN), fill with 0 or a reasonable value.
    # For this app, we assume that valid numeric input will be provided.

    # Apply Label Encoding for binary columns using loaded encoders
    # Ensure only columns expected by encoders are transformed
    for col, le in label_encoders.items():
        if col == 'Churn': # 'Churn' is the target, not a feature for new input
            continue
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                st.warning(f"Value '{df[col].iloc[0]}' for {col} not seen during training. This might cause issues.")
                # For robustness, we could try to handle unseen categories, but for now a warning is sufficient.

    # Apply One Hot Encoding for multi-category columns
    one_hot_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']
    df = pd.get_dummies(df, columns=[col for col in one_hot_cols if col in df.columns], dtype=int)

    # Align columns with the training data's feature_columns
    # Add missing columns (from training data) and fill with 0
    missing_cols = set(feature_columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0

    # Drop columns not in the training data
    extra_cols = set(df.columns) - set(feature_columns)
    df = df.drop(columns=list(extra_cols))

    # Ensure the order of columns is the same as during training
    df = df[feature_columns]

    return df

# --- Streamlit App Layout ---
st.title('Predicción de Churn de Clientes de Telecomunicaciones')
st.write('Ingrese los detalles del cliente para predecir si churneará o no.')

# Input fields for features
with st.form('churn_prediction_form'):
    st.header('Datos del Cliente')

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Género', ['Male', 'Female'])
        SeniorCitizen = st.selectbox('Ciudadano Mayor', [0, 1], format_func=lambda x: 'Sí' if x==1 else 'No')
        Partner = st.selectbox('Socio', ['Yes', 'No'])
        Dependents = st.selectbox('Dependientes', ['Yes', 'No'])
        PhoneService = st.selectbox('Servicio Telefónico', ['Yes', 'No'])
        MultipleLines = st.selectbox('Múltiples Líneas', ['No phone service', 'No', 'Yes'])
        InternetService = st.selectbox('Servicio de Internet', ['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox('Seguridad Online', ['No', 'Yes', 'No internet service'])
        OnlineBackup = st.selectbox('Copia de Seguridad Online', ['No', 'Yes', 'No internet service'])
        DeviceProtection = st.selectbox('Protección de Dispositivo', ['No', 'Yes', 'No internet service'])

    with col2:
        TechSupport = st.selectbox('Soporte Técnico', ['No', 'Yes', 'No internet service'])
        StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        StreamingMovies = st.selectbox('Streaming Películas', ['No', 'Yes', 'No internet service'])
        Contract = st.selectbox('Contrato', ['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.selectbox('Facturación Electrónica', ['Yes', 'No'])
        PaymentMethod = st.selectbox('Método de Pago',
                                    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        tenure = st.slider('Antigüedad (meses)', 0, 72, 1)
        MonthlyCharges = st.number_input('Cargos Mensuales', min_value=0.0, value=20.0)
        TotalCharges = st.number_input('Cargos Totales', min_value=0.0, value=20.0)
        # Ensure TotalCharges is not less than MonthlyCharges for tenure > 0, if needed for validation
        if tenure == 0 and TotalCharges != 0.0:
            st.warning("TotalCharges should be 0 if tenure is 0.")
        elif tenure > 0 and TotalCharges < MonthlyCharges:
            st.warning("TotalCharges should generally be greater than or equal to MonthlyCharges for tenure > 0.")

    submitted = st.form_submit_button('Predecir Churn')

    if submitted:
        # Create a DataFrame from inputs
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }])

        # Preprocess input data
        processed_input = preprocess_input(input_data, label_encoders, feature_columns)

        # Make prediction
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

        # Decode prediction
        churn_le = label_encoders['Churn']
        decoded_prediction = churn_le.inverse_transform(prediction)[0]

        st.subheader('Resultado de la Predicción:')
        if decoded_prediction == 'Yes':
            st.error(f"El cliente probablemente hará CHURN (abandono). Probabilidad: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.success(f"El cliente probablemente NO hará CHURN. Probabilidad: {prediction_proba[0][0]*100:.2f}%")

st.write("Para ejecutar esta aplicación Streamlit, guarde este código como `streamlit_app.py` y ejecute `streamlit run streamlit_app.py` en su terminal.")
