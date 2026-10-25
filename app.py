
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical

st.set_page_config(
    page_title="MindBalance | Capstone Project",
    page_icon="🧠",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# -------------------- CUSTOM ACTIVATION --------------------
@tf.keras.utils.register_keras_serializable()
def custom_activation(x):
    return tf.keras.activations.tanh(x) + (
        tf.keras.activations.relu(x) * 1.5
    )


# -------------------- SESSION STATE --------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "encoder" not in st.session_state:
    st.session_state.encoder = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None
if "numeric_columns" not in st.session_state:
    st.session_state.numeric_columns = None
if "history" not in st.session_state:
    st.session_state.history = None
if "test_results" not in st.session_state:
    st.session_state.test_results = None


# -------------------- SIDEBAR --------------------
st.sidebar.title("🧠 MindBalance")
st.sidebar.caption("Social Media Usage & Emotional Well-Being")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "Exploratory Data Analysis",
        "Train & Evaluate Model",
        "Predict Emotion",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Capstone project using exploratory data analysis "
    "and a GRU neural network."
)


# -------------------- DATA UPLOAD --------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file, on_bad_lines="skip")


def prepare_frame(df):
    """Apply the notebook's missing-value and one-hot encoding steps."""
    df = df.dropna().copy()

    categorical = [
        c for c in ["Gender", "Platform"] if c in df.columns
    ]

    df = pd.get_dummies(
        df,
        columns=categorical,
        drop_first=True,
        dtype=int
    )
    return df


def make_gru_model(input_features, num_classes):
    model = Sequential([
        tf.keras.layers.Input(shape=(1, input_features)),
        GRU(32, activation="relu", return_sequences=True),
        Dropout(0.2),
        GRU(
            128,
            activation=custom_activation,
            return_sequences=True
        ),
        Dropout(0.2),
        GRU(64, activation=custom_activation),
        Dropout(0.2),
        Dense(32, activation="tanh"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# -------------------- HOME --------------------
if page == "Home":
    st.title("Social Media Usage & Emotional Well-Being")
    st.subheader("Explore data. Understand patterns. Predict emotions.")

    st.markdown(
        """
        Welcome to **MindBalance**, an interactive dashboard for the
        capstone project investigating the relationship between social
        media usage patterns and users' dominant emotions.

        The project uses exploratory data analysis and a
        **Gated Recurrent Unit (GRU)** neural network to classify
        dominant emotions from the available user attributes.
        """
    )

    st.info(
        "Upload the train, test and validation CSV files from your "
        "dataset in the Dataset Overview section to begin."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Analysis", "EDA")
    c2.metric("Deep Learning", "GRU")
    c3.metric("Target", "Dominant Emotion")

    st.markdown("### Project workflow")

    cols = st.columns(5)
    steps = [
        ("1", "Load data"),
        ("2", "Explore data"),
        ("3", "Preprocess"),
        ("4", "Train GRU"),
        ("5", "Evaluate")
    ]

    for col, (number, label) in zip(cols, steps):
        with col:
            st.markdown(f"### {number}")
            st.caption(label)

    st.markdown("---")
    st.caption(
        "Educational prototype only. Predicted emotions are not "
        "clinical mental-health assessments."
    )


# -------------------- DATASET OVERVIEW --------------------
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write(
        "Upload the dataset splits used in your Jupyter notebook."
    )

    train_file = st.file_uploader(
        "Upload train.csv",
        type="csv",
        key="train"
    )
    test_file = st.file_uploader(
        "Upload test.csv",
        type="csv",
        key="test"
    )
    val_file = st.file_uploader(
        "Upload val.csv",
        type="csv",
        key="val"
    )

    if train_file:
        try:
            train_df = load_csv(train_file)
            st.session_state.train_df = train_df

            st.success("Training dataset loaded successfully.")

            a, b, c = st.columns(3)
            a.metric("Training records", f"{len(train_df):,}")
            b.metric("Number of columns", len(train_df.columns))
            c.metric(
                "Missing values",
                int(train_df.isnull().sum().sum())
            )

            st.markdown("### Data preview")
            st.dataframe(train_df.head(10), use_container_width=True)

            left, right = st.columns(2)

            with left:
                st.markdown("### Column data types")
                dtype_df = pd.DataFrame({
                    "Column": train_df.columns,
                    "Data type": train_df.dtypes.astype(str).values
                })
                st.dataframe(dtype_df, use_container_width=True)

            with right:
                st.markdown("### Missing values")
                missing = train_df.isnull().sum()
                missing = missing[missing > 0]
                if len(missing):
                    st.dataframe(
                        missing.rename("Missing count"),
                        use_container_width=True
                    )
                else:
                    st.success("No missing values found.")

            if test_file:
                test_df = load_csv(test_file)
                st.session_state.test_df = test_df
                st.metric("Test records", f"{len(test_df):,}")

            if val_file:
                val_df = load_csv(val_file)
                st.session_state.val_df = val_df
                st.metric("Validation records", f"{len(val_df):,}")

        except Exception as e:
            st.error(f"Unable to read the dataset: {e}")
    else:
        st.warning("Upload train.csv to see the dataset overview.")


# -------------------- EDA --------------------
elif page == "Exploratory Data Analysis":
    st.title("📈 Exploratory Data Analysis")

    if "train_df" not in st.session_state:
        st.warning(
            "Please upload train.csv on the Dataset Overview page first."
        )
        st.stop()

    df = st.session_state.train_df.copy()

    st.markdown("### Summary statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if "Age" in df.columns:
        st.markdown("### Age distribution")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(
            df["Age"].dropna(),
            bins=25,
            kde=True,
            ax=ax
        )
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        plt.close(fig)

    if "Gender" in df.columns:
        st.markdown("### Gender distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x="Gender", ax=ax)
        ax.set_title("Gender Distribution")
        st.pyplot(fig)
        plt.close(fig)

    if (
        "Platform" in df.columns
        and "Daily_Usage_Time (minutes)" in df.columns
    ):
        st.markdown("### Average daily usage by platform")
        platform_usage = (
            df.groupby("Platform")["Daily_Usage_Time (minutes)"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(platform_usage)

    if "Dominant_Emotion" in df.columns:
        st.markdown("### Dominant emotion distribution")
        emotion_counts = df["Dominant_Emotion"].value_counts()

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(
            x=emotion_counts.index,
            y=emotion_counts.values,
            ax=ax
        )
        ax.set_xlabel("Dominant Emotion")
        ax.set_ylabel("Number of records")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
        plt.close(fig)

    if len(numeric_cols) > 1:
        st.markdown("### Numerical feature correlation")
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)

    if categorical_cols:
        with st.expander("Explore another categorical feature"):
            selected_cat = st.selectbox(
                "Select a feature",
                categorical_cols
            )
            st.bar_chart(df[selected_cat].value_counts())


# -------------------- TRAIN & EVALUATE --------------------
elif page == "Train & Evaluate Model":
    st.title("🧠 Train & Evaluate GRU Model")

    required = ["train_df", "test_df", "val_df"]
    if not all(k in st.session_state for k in required):
        st.warning(
            "Upload train.csv, test.csv and val.csv on the "
            "Dataset Overview page first."
        )
        st.stop()

    train_df = st.session_state.train_df.copy()
    test_df = st.session_state.test_df.copy()
    val_df = st.session_state.val_df.copy()

    target = "Dominant_Emotion"

    if not all(target in d.columns for d in [train_df, test_df, val_df]):
        st.error(f"All three files must contain the '{target}' column.")
        st.stop()

    epochs = st.slider("Training epochs", 1, 100, 20)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)

    st.caption(
        "The notebook uses up to 300 epochs. Start with fewer epochs "
        "to test the app before running longer training."
    )

    if st.button("Train GRU Model", type="primary"):
        with st.spinner("Preprocessing data and training the model..."):
            try:
                # Remove missing values as in the notebook
                train_df = train_df.dropna().copy()
                test_df = test_df.dropna().copy()
                val_df = val_df.dropna().copy()

                # Encode labels consistently across the three splits
                encoder = LabelEncoder()
                encoder.fit(pd.concat([
                    train_df[target],
                    test_df[target],
                    val_df[target]
                ]).astype(str))

                y_train = encoder.transform(train_df[target].astype(str))
                y_test = encoder.transform(test_df[target].astype(str))
                y_val = encoder.transform(val_df[target].astype(str))

                # One-hot encode Gender and Platform, matching notebook
                X_train_df = prepare_frame(train_df.drop(columns=[target]))
                X_test_df = prepare_frame(test_df.drop(columns=[target]))
                X_val_df = prepare_frame(val_df.drop(columns=[target]))

                # Align test and validation columns to training columns
                feature_columns = X_train_df.columns.tolist()
                X_test_df = X_test_df.reindex(
                    columns=feature_columns, fill_value=0
                )
                X_val_df = X_val_df.reindex(
                    columns=feature_columns, fill_value=0
                )

                numeric_columns = X_train_df.select_dtypes(
                    include=np.number
                ).columns.tolist()

                scaler = StandardScaler()
                X_train = scaler.fit_transform(
                    X_train_df[numeric_columns]
                )
                X_test = scaler.transform(
                    X_test_df[numeric_columns]
                )
                X_val = scaler.transform(
                    X_val_df[numeric_columns]
                )

                # GRU expects 3-dimensional input
                X_train = np.expand_dims(X_train, axis=1)
                X_test = np.expand_dims(X_test, axis=1)
                X_val = np.expand_dims(X_val, axis=1)

                num_classes = len(encoder.classes_)
                y_train_oh = to_categorical(y_train, num_classes)
                y_test_oh = to_categorical(y_test, num_classes)
                y_val_oh = to_categorical(y_val, num_classes)

                model = make_gru_model(
                    X_train.shape[2],
                    num_classes
                )

                progress = st.progress(0)
                status = st.empty()

                class ProgressCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress.progress((epoch + 1) / epochs)
                        status.write(
                            f"Epoch {epoch + 1}/{epochs} — "
                            f"accuracy: {logs.get('accuracy', 0):.4f}, "
                            f"validation accuracy: "
                            f"{logs.get('val_accuracy', 0):.4f}"
                        )

                history = model.fit(
                    X_train,
                    y_train_oh,
                    validation_data=(X_val, y_val_oh),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[ProgressCallback()]
                )

                test_loss, test_accuracy = model.evaluate(
                    X_test, y_test_oh, verbose=0
                )

                predictions = np.argmax(
                    model.predict(X_test, verbose=0),
                    axis=1
                )

                report = classification_report(
                    y_test,
                    predictions,
                    labels=np.arange(num_classes),
                    target_names=encoder.classes_,
                    output_dict=True,
                    zero_division=0
                )

                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.encoder = encoder
                st.session_state.feature_columns = feature_columns
                st.session_state.numeric_columns = numeric_columns
                st.session_state.history = history.history
                st.session_state.test_results = {
                    "loss": test_loss,
                    "accuracy": test_accuracy,
                    "report": report,
                    "y_test": y_test,
                    "predictions": predictions,
                    "classes": encoder.classes_
                }

                st.success("Training and evaluation completed!")

            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state.model is not None:
        results = st.session_state.test_results

        a, b = st.columns(2)
        a.metric("Test accuracy", f"{results['accuracy']:.2%}")
        b.metric("Test loss", f"{results['loss']:.4f}")

        history = st.session_state.history

        st.markdown("### Training history")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.write("Accuracy")
            st.line_chart(pd.DataFrame({
                "Training": history["accuracy"],
                "Validation": history["val_accuracy"]
            }))

        with chart_col2:
            st.write("Loss")
            st.line_chart(pd.DataFrame({
                "Training": history["loss"],
                "Validation": history["val_loss"]
            }))

        st.markdown("### Classification report")
        st.dataframe(
            pd.DataFrame(results["report"]).T,
            use_container_width=True
        )

        st.markdown("### Confusion matrix")
        cm = confusion_matrix(
            results["y_test"],
            results["predictions"],
            labels=np.arange(len(results["classes"]))
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=results["classes"],
            yticklabels=results["classes"],
            ax=ax
        )
        ax.set_xlabel("Predicted emotion")
        ax.set_ylabel("Actual emotion")
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Download trained model",
            data=__import__("io").BytesIO(
                _save_model_bytes(st.session_state.model)
            ).getvalue() if False else b"",
            file_name="gru_emotion_model.h5",
            disabled=True,
            help="Use the Save Model button below to create the model file."
        )

        if st.button("Save model to current folder"):
            st.session_state.model.save("gru_emotion_model.keras")
            st.success("Saved as gru_emotion_model.keras")


# -------------------- PREDICT EMOTION --------------------
elif page == "Predict Emotion":
    st.title("🔮 Predict Dominant Emotion")

    model = st.session_state.model
    scaler = st.session_state.scaler
    encoder = st.session_state.encoder

    if model is None:
        st.warning(
            "Train the GRU model first using the Train & Evaluate Model page."
        )
        st.stop()

    st.write(
        "Enter values for the features used by the model. "
        "Predictions describe dataset emotion categories, not a "
        "clinical diagnosis."
    )

    train_df = st.session_state.train_df.copy()
    target = "Dominant_Emotion"
    raw_features = train_df.dropna().drop(columns=[target])

    # Ask for values using the original feature columns
    input_values = {}

    with st.form("prediction_form"):
        cols = st.columns(2)

        for i, col_name in enumerate(raw_features.columns):
            with cols[i % 2]:
                if col_name in ["Gender", "Platform"]:
                    choices = sorted(
                        raw_features[col_name].dropna().astype(str).unique()
                    )
                    input_values[col_name] = st.selectbox(
                        col_name, choices
                    )

                elif pd.api.types.is_numeric_dtype(
                    raw_features[col_name]
                ):
                    median_value = float(
                        raw_features[col_name].median()
                    )
                    input_values[col_name] = st.number_input(
                        col_name,
                        value=median_value
                    )

                else:
                    choices = sorted(
                        raw_features[col_name].dropna().astype(str).unique()
                    )
                    input_values[col_name] = st.selectbox(
                        col_name, choices
                    )

        submitted = st.form_submit_button(
            "Predict Dominant Emotion",
            type="primary"
        )

    if submitted:
        try:
            input_df = pd.DataFrame([input_values])
            encoded_input = pd.get_dummies(
                input_df,
                columns=[
                    c for c in ["Gender", "Platform"]
                    if c in input_df.columns
                ],
                drop_first=True,
                dtype=int
            )

            encoded_input = encoded_input.reindex(
                columns=st.session_state.feature_columns,
                fill_value=0
            )

            numeric_columns = st.session_state.numeric_columns
            scaled_input = scaler.transform(
                encoded_input[numeric_columns]
            )
            gru_input = np.expand_dims(scaled_input, axis=1)

            probabilities = model.predict(gru_input, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            predicted_emotion = encoder.inverse_transform(
                [predicted_index]
            )[0]

            st.success(f"Predicted dominant emotion: {predicted_emotion}")
            st.metric(
                "Model probability for predicted class",
                f"{probabilities[predicted_index]:.2%}"
            )

            probability_df = pd.DataFrame({
                "Emotion": encoder.classes_,
                "Probability": probabilities
            }).set_index("Emotion")

            st.markdown("### Class probabilities")
            st.bar_chart(probability_df)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -------------------- ABOUT --------------------
elif page == "About Project":
    st.title("ℹ️ About MindBalance")

    st.markdown(
        """
        ### Project title
        **Social Media Usage and Emotional Well-Being**

        ### Objective
        To explore social media usage patterns and develop a GRU
        neural network that predicts a user's dominant emotion using
        the attributes available in the dataset.

        ### Workflow
        1. Load the training, test and validation datasets.
        2. Explore age, gender, platform usage and dominant emotion.
        3. Drop rows with missing values.
        4. One-hot encode Gender and Platform.
        5. Standardize numerical features with `StandardScaler`.
        6. Train a GRU-based multi-class classifier.
        7. Evaluate using test accuracy, loss, classification report
           and confusion matrix.

        ### Technology stack
        - Python
        - Pandas and NumPy
        - Matplotlib and Seaborn
        - Scikit-learn
        - TensorFlow / Keras
        - Streamlit

        ### Limitations
        - The model predicts labels represented in the dataset; it
          does not diagnose mental-health conditions.
        - GRU is designed for sequential data, but this notebook
          reshapes each feature row into a sequence of length one.
        - Results depend on dataset quality, feature selection,
          model configuration and the train/test split.
        """
    )

    st.caption("Capstone project dashboard | MindBalance")
