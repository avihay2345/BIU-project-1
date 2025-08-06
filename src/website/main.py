import numpy
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import chardet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import io

st.set_page_config(page_title="Automatic EDA", layout="wide")

st.title("📊 Automatic EDA Generator")

# Upload interface
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    # Step 1: Read raw bytes from uploaded file
    raw_data = uploaded_file.read()

    result = chardet.detect(raw_data)
    detected_encoding = result['encoding']
    confidence = result['confidence']

    st.info(f"Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")

    uploaded_file.seek(0)  # reset file pointer
    try:
        df = pd.read_csv(uploaded_file, encoding=detected_encoding)
        df = df[0:200]
        # added this segment because i worked on a csv with lots of columns that look irrelevant
        threshold = 0.9  # Remove columns with more than 90% zeros
        numeric_cols = df.select_dtypes(include='number')
        zero_fraction = (numeric_cols == 0).sum() / len(df)
        cols_to_drop = zero_fraction[zero_fraction > threshold].index
        df = df.drop(columns=cols_to_drop)
        st.success("File loaded successfully!")
        st.header("1. Dataset Overview")
        st.write("### Preview")
        # Let user pick which columns to show
        all_columns = df.columns.tolist()
        default_cols = all_columns  # You can also use a subset here
        selected_columns = st.multiselect("Select columns to view", options=all_columns, default=default_cols)

        # Show selected columns only
        if selected_columns:
            st.dataframe(df[selected_columns].head())
        else:
            st.info("No columns selected. Use the selector above to choose what to display.")

        st.write("### Shape")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        st.write("### Data Types")
        st.dataframe(df.dtypes.reset_index().rename(columns={0: "Data Type", "index": "Column"}))

        st.header("2. Missing Data")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing Values']
        missing_df['% Missing'] = (missing_df['Missing Values'] / len(df)) * 100
        st.dataframe(missing_df)

        show_missing_plot = st.checkbox("Show Missing Value Bar Plot")
        if show_missing_plot:
            fig, ax = plt.subplots()
            sns.barplot(data=missing_df, x='Column', y='Missing Values', ax=ax)
            plt.xticks(rotation=-90)
            st.pyplot(fig)

        st.header("3. Summary Statistics")
        st.write("### Numeric Columns")
        st.dataframe(df.describe().T)

        st.write("### Categorical Columns")
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if len(cat_cols) > 0:
            st.dataframe(df[cat_cols].describe().T)

        st.header("4. Visualizations")

        st.subheader("Distributions & Box Plots")
        st.write("### Column Data Types")
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}))
        num_cols = df.select_dtypes(include='number').columns.tolist()
        selected_num_col = st.selectbox("Select numeric column", num_cols)
        if selected_num_col:
            col1, col2 = st.columns(2)

            with col1:
                st.write("Histogram")
                fig = px.histogram(df, x=selected_num_col)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("Box Plot")
                fig = px.box(df, y=selected_num_col)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        if st.checkbox("Show Correlation Matrix"):
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        st.subheader("Pair Plot (Seaborn)")
        if st.checkbox("Show Pair Plot (slower for large datasets)"):
            if len(num_cols) <= 10:
                hue_option = st.selectbox("Optional: Color by (hue)", [None] + cat_cols)

                try:
                    fig = sns.pairplot(
                        df,
                        vars=num_cols,
                        hue=hue_option if hue_option else None,
                        plot_kws={"s": 20, "alpha": 0.6},  # dot size and transparency
                        diag_kws={"fill": True}
                    )

                    # Format all subplots for better readability
                    for ax in fig.axes.flatten():
                        if ax:  # not None
                            ax.tick_params(labelsize=8)
                            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
                            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Failed to generate pair plot: {e}")
            else:
                st.warning("Too many numeric columns for pair plot (limit: 10).")

        st.subheader("Prediction (KNN)")
        rand_sample = 0
        if st.button('press to choose a random sample'):
            rand_sample = np.random.randint(len(df))
        st.write(f'line number {rand_sample} was selected:')
        st.dataframe(df.iloc[rand_sample])
        k = st.slider('select number of neighbors:', min_value=1, max_value=13, value=7)
        col_to_predict = st.selectbox("Select column to predict", cat_cols)
        subset = df.drop(rand_sample)
        subset_labels = subset[col_to_predict]
        if col_to_predict:
            subset = df.drop(rand_sample)
            subset_labels = subset[col_to_predict]
            selected_columns = st.multiselect("select (numeric) features to consider", options=num_cols,
                                              default=num_cols)
            if selected_columns:
                subset = subset[selected_columns]
                knn = KNeighborsClassifier(n_neighbors=k)
                scaler = StandardScaler()
                train_data_scaled = scaler.fit_transform(subset)
                knn.fit(subset, df.drop(rand_sample)[col_to_predict])
                sample = subset.iloc[[rand_sample]]
                sample_scaled = scaler.transform(sample)
                prediction = knn.predict(sample_scaled)
                distances, indices = knn.kneighbors(sample_scaled, return_distance=True)
                indices = indices.flatten()
                neighbors = subset.iloc[indices]
                neighbors[col_to_predict] = subset_labels.iloc[indices]
                st.write(f'the {k} nearest neighbors were:')
                st.dataframe(neighbors)
                st.write(f'the prediction was {prediction} and the real answer was {df.iloc[rand_sample][col_to_predict]}')

    except Exception as e:
        st.error(f"Failed to load CSV: {e}")


else:
    st.info("👈 Please upload a CSV file to begin.")


