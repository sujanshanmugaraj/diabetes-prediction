import streamlit as st
import plotly.express as px
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

data2 = pd.read_csv("C:\\Users\\Ritanya\\Downloads\\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

st.set_page_config(page_title="Diabetes", page_icon=":bar_chart:", layout="wide")
st.title(" :bar_chart: DIABETES PREDICTION")
st.sidebar.header('Input Your Health Metrics')

age = st.sidebar.slider('Age', min_value=0, max_value=120, value=50)
bmi = st.sidebar.slider('BMI', min_value=0.0, max_value=100.0, value=25.0)
gen_hlth = st.sidebar.slider('General Health (1-5)', min_value=1, max_value=5, value=3)
ment_hlth = st.sidebar.slider('Mental Health (1-5)', min_value=1, max_value=5, value=3)
phys_hlth = st.sidebar.slider('Physical Health (1-30)', min_value=1, max_value=30, value=15)
education = st.sidebar.slider('Education (1-10)', min_value=1, max_value=10, value=3)
income = st.sidebar.slider('Income (1-10)', min_value=1, max_value=10, value=3)

yes_no_variables = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity','Fruits','Veggies','HvyAlcoholConsump', 'AnyHealthcare','NoDocbcCost','DiffWalk','Sex']
input_values = {}
for var in yes_no_variables:
    input_values[var] = st.sidebar.checkbox(f'{var} (Yes/No)')


for var, value in input_values.items():
    input_values[var] = 1 if value else 0


x1 = data2.drop('Diabetes_binary',axis = 1)
y1 = data2['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3, random_state = 4)
lr2 = LogisticRegression(max_iter = 1000000)
lr2.fit(X_train, y_train.ravel())


pred_proba = lr2.predict_proba([[input_values['HighBP'],input_values['HighChol'],input_values['CholCheck'],bmi,input_values['Smoker'],input_values['Stroke'],input_values['HeartDiseaseorAttack'],input_values['PhysActivity'],input_values['Fruits'],input_values['Veggies'],input_values['HvyAlcoholConsump'],input_values['AnyHealthcare'],input_values['NoDocbcCost'],gen_hlth , ment_hlth , phys_hlth , input_values['DiffWalk'],input_values['Sex'],age,education,income]])
st.subheader('Predicted Probability of Developing Diabetes')
probability = pred_proba[0][1]

if probability < 0.3:
    color = 'green'
elif probability < 0.7:
    color = 'orange'
else:
    color = 'red'

bar_width = probability


df = pd.DataFrame({'Probability': [probability], 'Color': [color]})


fig = px.bar(df, x='Probability', color='Color', color_discrete_map={'green': 'green', 'orange': 'orange', 'red': 'red'},
             labels={'Probability': 'Probability of Diabetes'}, height=150, width=300)
fig.update_yaxes(visible=False)
fig.update_layout(showlegend=False)
fig.update_xaxes(range=[0, 1.05]) 
fig.update_traces(marker=dict(line=dict(width=bar_width)))
st.plotly_chart(fig, use_container_width=True)

y_pred = lr2.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)


accuracy_percentage = lr_accuracy * 100
accuracy_rating = '⭐⭐⭐⭐'
st.markdown(f'<p style="font-size:20px">Accuracy : {accuracy_rating} ({accuracy_percentage:.2f}%)</p>', unsafe_allow_html=True)
st.markdown(f'<p style="font-size:20px">Probability of having diabetes: {probability:.2f}</p>', unsafe_allow_html=True)

#################################################################
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>GENERAL ANALYSIS</h2>", unsafe_allow_html=True)
target_variable = 'Diabetes_binary'
feature_variables = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
correlations = data2[feature_variables].corrwith(data2[target_variable])

st.subheader('Correlation with Target Variable (Diabetes_binary)')
plt.figure(figsize=(10, 8))
sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Feature Variable')
plt.title('Correlation with Target Variable')
st.pyplot(plt)

st.write("The correlation between the target variable (diabetes) and all the feature variables are shown . \nHaving High BP and General Health are significant factors which leads to diabetes.")

###########################################################
st.markdown("<hr/>", unsafe_allow_html=True)
diabetes_df = data2.copy()
selected_feature = st.selectbox('Select a feature variable', feature_variables)
feature_sum_df = diabetes_df.groupby(selected_feature, as_index=False)['Diabetes_binary'].sum()

st.subheader(f'Bar Chart for {selected_feature}')
fig = px.bar(feature_sum_df, x=selected_feature, y='Diabetes_binary', text='Diabetes_binary', template='seaborn')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_xaxes(title_text=f'{selected_feature} (0:NO , 1: YES)')
fig.update_yaxes(title_text='No. of people')
st.plotly_chart(fig, use_container_width=True)

st.write("The Bar Chart tells how many people have diabetes with respect to feature variables \nFor example , we can infer that 8.7k people who do not have high BP has diabetes and 27k people who have High BP have diabetes.")
###########################################################
st.markdown("<hr/>", unsafe_allow_html=True)
data2 = pd.read_csv("C:\\Users\\Ritanya\\Downloads\\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

income_bins = [0, 2, 5, 8]
income_labels = ['Less than $10,000', 'Less than $35,000', '$35,000 - $75,000 or more']

data2["Income Category"] = pd.cut(data2["Income"], bins=income_bins, labels=income_labels, include_lowest=True)

diabetes_by_income = data2.groupby("Income Category")["Diabetes_binary"].value_counts().unstack().fillna(0)

diabetes_by_income["Total"] = diabetes_by_income.sum(axis=1)
diabetes_by_income["Diabetes Percentage"] = diabetes_by_income[1] / diabetes_by_income["Total"]
diabetes_by_income = diabetes_by_income.reset_index()

diabetes_by_income = diabetes_by_income.dropna(subset=["Diabetes Percentage"])
diabetes_by_income = diabetes_by_income[diabetes_by_income["Diabetes Percentage"] != 0]

#############################################################
age_categories = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80 or older"
}


data2["Age Category"] = data2["Age"].map(age_categories)
diabetes_by_age = data2.groupby("Age Category")["Diabetes_binary"].value_counts().unstack().fillna(0)
diabetes_by_age["Total"] = diabetes_by_age.sum(axis=1)
diabetes_by_age["Diabetes Percentage"] = diabetes_by_age[1] / diabetes_by_age["Total"]
diabetes_by_age = diabetes_by_age.reset_index()
diabetes_by_age = diabetes_by_age.dropna(subset=["Diabetes Percentage"])
diabetes_by_age = diabetes_by_age[diabetes_by_age["Diabetes Percentage"] != 0]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Diabetes by Income Level")
    fig_income = px.pie(diabetes_by_income, values="Diabetes Percentage", names="Income Category", 
                        title="Income Level",
                        hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_income.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
    st.plotly_chart(fig_income, use_container_width=True)

with col2:
    st.subheader("Distribution of Diabetes by Age Group")
    fig_age = px.pie(diabetes_by_age, values="Diabetes Percentage", names="Age Category", 
                     title="Age Group",
                     hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_age.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
    st.plotly_chart(fig_age, use_container_width=True)

##############################################################################
diabetes_by_sex = data2.groupby("Sex")["Diabetes_binary"].value_counts().unstack().fillna(0)
diabetes_by_sex["Total"] = diabetes_by_sex.sum(axis=1)
diabetes_by_sex["Diabetes Percentage"] = diabetes_by_sex[1] / diabetes_by_sex["Total"]
diabetes_by_sex = diabetes_by_sex.reset_index()
diabetes_by_sex = diabetes_by_sex.dropna(subset=["Diabetes Percentage"])
diabetes_by_sex = diabetes_by_sex[diabetes_by_sex["Diabetes Percentage"] != 0]
diabetes_by_sex["Sex"] = diabetes_by_sex["Sex"].map({0: "Female", 1: "Male"})

education_categories = {
    1: "Never attended school or only kindergarten",
    2: "Elementary",
    3: "Some high school",
    4: "High school graduate or GED",
    5: "Some college or tech school",
    6: "College graduate"
}

data2["Education Level"] = data2["Education"].map(education_categories)
diabetes_by_education = data2.groupby("Education Level")["Diabetes_binary"].value_counts().unstack().fillna(0)
diabetes_by_education["Total"] = diabetes_by_education.sum(axis=1)
diabetes_by_education["Diabetes Percentage"] = diabetes_by_education[1] / diabetes_by_education["Total"]
diabetes_by_education = diabetes_by_education.reset_index()
diabetes_by_education = diabetes_by_education.dropna(subset=["Diabetes Percentage"])
diabetes_by_education = diabetes_by_education[diabetes_by_education["Diabetes Percentage"] != 0]

with col1:
    st.subheader("Distribution of Diabetes by Gender")
    fig_sex = px.pie(diabetes_by_sex, values="Diabetes Percentage", names="Sex", title="Distribution of Diabetes by Sex",
                 hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_sex.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
    st.plotly_chart(fig_sex, use_container_width=True)

with col2:
    st.subheader("Distribution of Diabetes by Education")
    fig_education = px.pie(diabetes_by_education, values="Diabetes Percentage", names="Education Level",
                       title="Distribution of Diabetes by Education Level",
                       hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_education.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
    st.plotly_chart(fig_education, use_container_width=True)

st.write("These plots provide insights into the distribution of diabetes across various demographic factors such as income level, age group, gender, and education level. They visualize the proportion of individuals with diabetes within each category, highlighting potential associations between demographic factors and the likelihood of having diabetes.")
####################################################################################################3

st.markdown("<hr/>", unsafe_allow_html=True)
binary_variable = "Diabetes_binary"
feature_variables = ['HighBP', 'HighChol', 'CholCheck','Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                     'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 
                     'DiffWalk', 'Sex']

grouped_data_all = []
for feature_variable in feature_variables:
    grouped_data = data2.groupby([feature_variable, binary_variable]).size().unstack(fill_value=0)
    grouped_data = grouped_data.div(grouped_data.sum(axis=1), axis=0)
    grouped_data.reset_index(inplace=True)
    grouped_data["Feature Variable"] = feature_variable
    grouped_data_all.append(grouped_data)
grouped_data_all = pd.concat(grouped_data_all)

fig = px.bar(grouped_data_all, 
             x="Feature Variable",
             y=[0, 1], 
             barmode="stack",
             color_discrete_sequence=px.colors.qualitative.Set1,
             title=f"Distribution of {binary_variable} Across Feature Variables",
             labels={"value": "Proportion", "variable": binary_variable},
             hover_name="Feature Variable")


fig.update_layout(xaxis_title="Feature Variable", yaxis_title="Proportion", legend_title=binary_variable)
st.plotly_chart(fig, use_container_width=True)

st.write("* Each bar in the plot corresponds to a specific category of the feature variable. For example, if the feature variable is HighBP (high blood pressure), each bar represents the proportion of individuals with and without high blood pressure who have diabetes.\n* The height of each colored segment within the bar represents the proportion of individuals with diabetes (or without diabetes) within that category of the feature variable. For example, if the blue segment is taller than the orange segment in the HighBP bar, it means that a higher proportion of individuals with high blood pressure have diabetes compared to those without high blood pressure.")
##########################################################

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader('Box Plots: Distribution of Numerical Features by Diabetes Status')
numerical_features = ['BMI', 'Age', 'Education', 'Income']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, num_feature in zip(axes.flatten(), numerical_features):
    sns.boxplot(x='Diabetes_binary', y=num_feature, data=data2, palette='Set2', ax=ax)
    ax.set_title(f'Distribution of {num_feature} by Diabetes Status')
    ax.set_xlabel('Diabetes Status')
    ax.set_ylabel(num_feature)

plt.tight_layout()
st.pyplot(fig)

st.write("The box plot shows the distribution of numerical features for individuals grouped by diabetes status. We can observe whether there are differences in BMI between diabetic and non-diabetic individuals. \nFor example, if the median BMI is higher for diabetic individuals compared to non-diabetic individuals, it suggests a potential association between BMI and diabetes.\nThere are no outliers in distribution of education and income")

#######################################################
st.markdown("<hr/>", unsafe_allow_html=True)
st.write("To mitigate the risk of developing diabetes, adopt a healthy lifestyle by maintaining a balanced diet rich in fruits, vegetables, and whole grains, while limiting processed foods, sugary snacks, and high-calorie beverages. Engage in regular physical activity, manage stress effectively, and prioritize regular health check-ups. Avoid smoking and excessive alcohol consumption. Making these lifestyle changes can significantly reduce the risk of diabetes and promote overall well-being.")