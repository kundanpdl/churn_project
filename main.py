## Importing streamlit
import streamlit as st

## Data processing and visualization.
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


## Machine learning libraries.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

## Evaluation of the models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


page_names = st.sidebar.selectbox("Go To: ", ["Dashboard", "Info", "Use Models"])

## Saving dataset name
dataset_name = "Churn_Modelling.csv"

## Creating dataframe
df = pd.read_csv(dataset_name)

if page_names == "Dashboard":
    ## Giving title of the Page
    st.title("Dashboard")
    st.write("""Dataset Preview""")
    st.write(df.head())
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(x='Gender', hue='Exited', data=df, ax=ax[0])
    ax[0].set_title("Churn (Exited) vs Gender")
    ax[0].set_xlabel("Gender")
    ax[0].set_ylabel("Count")
    ax[0].legend(title="Exited", labels=["No (0)", "Yes (1)"])
    
    sns.countplot(x='Tenure', hue='Exited', data=df, ax=ax[1])
    ax[1].set_title("Churn (Exited) vs Tenure")
    ax[1].set_xlabel("Tenure")
    ax[1].set_ylabel("Count")
    ax[1].legend(title="Exited", labels=["No (0)", "Yes (1)"])
    st.pyplot(fig)
    
elif page_names == "Info":
    ## Giving title of the Page
    st.title("Info")
    st.write("Info")
else:
    ## Giving title of the Page
    st.title("Churn Calculation Using Different ML Models")
    
    ## Creating sidebar with dropdown for different models
    model_names = st.sidebar.selectbox("Select ML Model", ("Logistic Regression", "Random Forest", "Support Vector Machine", "Gradient Boosting Classifier", "KNN"))
    st.write(model_names)

    label_encoder = LabelEncoder()

    ## Binary features for balancezero
    df['BalanceZero'] = (df['Balance'] == 0).astype(int)
    ## Creating age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65, 75, 85, 95], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95'])
    ## Creating balance to salary ratio
    df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
    ## Interaction feature between Numofproducts and isactivemember
    df['ProductUsage'] = df['NumOfProducts'] * df['IsActiveMember']
    ## Creating tenure group
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 2, 5, 7, 10], labels=['0-2', '3-5', '6-7', '8-10'])
        
    ## Label Encoding
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
    df['Male_Germany'] = df['Gender'] * df['Geography_Germany'] ## Because male = 1
    df['Male_Spain'] = df['Gender'] * df['Geography_Spain']


    ## One hot encoding
    df = pd.get_dummies(df, columns=['AgeGroup', 'TenureGroup'], drop_first=True)
        

    ## Getting X and y features from the dataset
    ## Feature Selection
    features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Geography_Germany','Geography_Spain', 'BalanceZero', 'BalanceToSalaryRatio', 'ProductUsage', 
                'Male_Germany', 'Male_Spain'] + [ col for col in df.columns if 'AgeGroup_' in col or 'TenureGroup_' in col]

    X = df[features]
    y = df['Exited']


    ## Creating function for different model selection
    def add_parameter_ui(model_name):
        params = dict()
        if model_name == 'Support Vector Machine':
            kernel = st.sidebar.selectbox("Select Kernel", ("linear", "poly", "rbf", "sigmoid"))
            params['kernel'] = kernel
        elif model_name == 'KNN':
            n_neighbors = st.sidebar.slider("Select n_neighbors: ", 1, 10)
            params['n_neighbors'] = n_neighbors
        elif model_name == 'Gradient Boosting Classifier':
            n_estimators = st.sidebar.slider("Select n_estimators: ", 5, 100)
            params['n_estimators'] = n_estimators
        elif model_name == "Random Forest":
            n_estimators_rf = st.sidebar.slider("Select n_estimators: ", 5, 100)
            params['n_estimators_rf'] = n_estimators_rf
        return params

    ## Running the function and displaying the parameters based on model chosen
    params = add_parameter_ui(model_names)


    ## Function to choose the ML model with ui-specified parameters
    def get_model(model_name, params):
        model = None
        if model_name == 'Support Vector Machine':
            model = SVC(kernel=params['kernel'], random_state=42) 
        elif model_name == 'KNN':
            model = knn_model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        elif model_name == 'Gradient Boosting Classifier':
            model = GradientBoostingClassifier(n_estimators=params['n_estimators'], random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=params['n_estimators_rf'], random_state=42)
        else:
            model = LogisticRegression(random_state=42)
        return model

    ## Get the chosen model with appropriate parameters from the ui
    model = get_model(model_names, params)

    ## Creating train and test dataset
    ## Random state is used so that you can access exactly same data in the future using the same number.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## Feature scaling. Making sure all data are in similar scale to increase model performance.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ## Model fitting and prediction
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ## Calculating accuracy
    ## Creating confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # st.write(conf_matrix)
    # st.write(class_report)
    # st.write(accuracy)
    
     # Confusion Matrix Plot
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Classification Report Table
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df.style.background_gradient(cmap="Blues").format(precision=2))

    # Accuracy Metric
    st.metric("Model Accuracy", f"{accuracy:.2%}")