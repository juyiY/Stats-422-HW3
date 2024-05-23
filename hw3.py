import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

# Load Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Title and description
st.markdown('<h1 style="font-size:43px;">KNN Demonstration with Iris Dataset</h1>', unsafe_allow_html=True)
st.markdown("##")
st.write('Hi there! My name is Juyi and I am creating a web app that demonstrates the K-Nearest Neighbors (KNN) algorithm using the Iris dataset. I have also attached the actual dataset below for reference.')

# Detailed Goal Description
st.markdown("""
### Goal:
This web app demonstrates the use of the K-Nearest Neighbors (KNN) algorithm to classify samples of iris flowers into one of three species based on flower measurements. It allows you to interactively explore how the KNN algorithm responds to different parameter settings, including the number of neighbors (k), the choice of distance metric, and the weighting approach. By adjusting these parameters and observing the changes in classification outcomes, you can gain insights into the behavior and sensitivity of the algorithm.
""")

# Detailed Data Source Description
st.markdown("""
### Data Source:
The Iris dataset used in this project is a well-known dataset in the fields of statistics and machine learning. It comprises 150 samples from three species of Iris flowers—setosa, versicolor, and virginica. Each sample is described by four quantitative features: the lengths and the widths of the sepals and petals, measured in centimeters.
""")

# Mapping numeric labels to species names for clarity
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['target'].map(species_mapping)

# Sidebar for user inputs
k = st.sidebar.slider('Select number of neighbors (k)', min_value=1, max_value=15, value=3)
metric = st.sidebar.selectbox('Select distance metric', options=['euclidean', 'manhattan', 'chebyshev'])
weights = st.sidebar.selectbox('Select weight type', options=['uniform', 'distance'])
x_axis_feature = st.sidebar.selectbox('Choose X-axis feature', data.feature_names, index=0)
y_axis_feature = st.sidebar.selectbox('Choose Y-axis feature', data.feature_names, index=1)
color_by = st.sidebar.selectbox('Color By', ['predicted_species', 'species', 'prediction_accuracy'])

# KNN model
model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
X = df[data.feature_names]
y = df['target']
model.fit(X, y)

# Make predictions and map to species names
df['prediction'] = model.predict(X)
df['predicted_species'] = df['prediction'].map(species_mapping)

# Define the color scheme for prediction accuracy
df['prediction_accuracy'] = df.apply(lambda row: 'Correct' if row['predicted_species'] == row['species'] else 'Incorrect', axis=1)
color_discrete_map = {'Correct': 'green', 'Incorrect': 'red', 'setosa': '#2CA02C', 'versicolor': '#FF7F0E', 'virginica': '#1F77B4'}  # Color-blind friendly

# Plotting
fig = px.scatter(
    df, 
    x=x_axis_feature, 
    y=y_axis_feature, 
    color=df[color_by],  # Make sure this is the correct column name
    symbol='species',
    labels={'species': 'True Species', color_by: color_by},
    color_discrete_map=color_discrete_map
)

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Show prediction accuracy summary
if 'prediction_accuracy' in color_by:
    correct_count = df['prediction_accuracy'].value_counts().get('Correct', 0)
    total = len(df)
    accuracy = (correct_count / total) * 100
    st.write(f"Prediction Accuracy: {accuracy:.2f}%")
    
# Show prediction accuracy summary
correct_count = df[df['prediction'] == df['target']].shape[0]
total = df.shape[0]
accuracy = (correct_count / total) * 100
st.markdown("""
### Prediction Accuracy:
""")
st.write(f"{accuracy:.2f}% --- based on {k} neighbors and {metric} metric with {weights} weighting.")
st.markdown("##")
st.markdown("""
### Actual labels for comparison:
""")
st.dataframe(df[[x_axis_feature, y_axis_feature, 'species', 'predicted_species']])

