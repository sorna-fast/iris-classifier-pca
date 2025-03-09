import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

st.title("Iris Dataset Classification with PCA")

iris = datasets.load_iris()
X = pd.DataFrame(iris.data,columns=iris.feature_names)
y = pd.Series(iris.target,name='species')

test_size = st.slider("Test Data Size Precentage" , 0.1 , 0.9 , 0.2)
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=test_size)


model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
st.markdown(f'<span style="color:green; font-weight:900;">Accuracy :</span>',unsafe_allow_html=True)
st.write(f"{accuracy:.2f}")

conf_matrix = confusion_matrix(y_test,y_pred)
st.markdown(f'<span style="color:green; font-weight:900;">Confusion Matrix :</span>',unsafe_allow_html=True)
st.write(conf_matrix)

report = classification_report(y_test,y_pred,target_names=iris.target_names)
st.markdown(f'<span style="color:green; font-weight:900;">Classification Report :</span>',unsafe_allow_html=True)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(X_pca,columns=["PCA1","PCA2"])
df_pca['species'] = y

st.markdown(f'<span style="color:green; font-weight:900;">PCA DataFrame :</span>',unsafe_allow_html=True)
st.dataframe(df_pca)

st.markdown(f'<span style="color:green; font-weight:900;">Scatter Plot using PCA :</span>',unsafe_allow_html=True)
fig , ax=plt.subplots(figsize=(10,6))
sns.scatterplot(x="PCA1",y="PCA2",hue=iris.target_names[df_pca["species"]],palette='Set1',data=df_pca,ax=ax)
ax.set_title("Dimensionality Raduction of Iris Dataset to 2 Components using PCA")
st.pyplot(fig)


st.sidebar.header("Enter Feature Values:")
sepal_length = st.sidebar.slider("Sepal Length",float(X['sepal length (cm)'].min()),float(X['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width",float(X['sepal width (cm)'].min()),float(X['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length",float(X['petal length (cm)'].min()),float(X['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width",float(X['petal width (cm)'].min()),float(X['petal width (cm)'].max()))

input_data = [[sepal_length,sepal_width,petal_length,petal_width]]
prediction = model.predict(input_data)
prediction_prob = model.predict_proba(input_data)
btn = st.sidebar.button("Predict")
if btn:
    st.sidebar.markdown(f'<span style="color:green; font-weight:900;">Prediction :</span>',unsafe_allow_html=True)
    st.sidebar.write(iris.target_names[prediction[0]])
    st.sidebar.markdown(f'<span style="color:green; font-weight:900;">Prediction Probabilites :</span>',unsafe_allow_html=True)
    for i, prob in enumerate(prediction_prob[0]):
        st.sidebar.write(f"{iris.target_names[i]}: {prob:.2f}")
