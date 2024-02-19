#Input the relevant libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Define the Streamlit app
def app():
    # Display the DataFrame with formatting
    st.title("Generate Dataset with Features and Classes")
    st.write(
        """This app generates  dataset with balanced classes 
        and informative features to facilitate exploration and analysis."""
        )
    displaysummary = False
    enabledownload = False
    # Add interactivity and customization options based on user feedback
    st.sidebar.header("Customization")
    if st.sidebar.checkbox("Include data summary?"):
        displaysummary = True
    else:
        displaysummary = False
    if st.sidebar.checkbox("Enable download?"):
        enabledownload = True
    else:
        enabledownload = False
    if st.button('Start'):
        # Data generation with balanced classes and informative features
        np.random.seed(42)  # For reproducibility
        num_samples = 100
        feature1 = np.random.normal(3, 2, size=num_samples)
        feature2 = np.random.normal(4, 3, size=num_samples)

        # Create informative classes based on features
        threshold = 8
        classes = (feature1 + 2 * feature2) > threshold
        labels = ['Class A' if label else 'Class B' for label in classes]

        # Combine features and labels into a DataFrame
        data = pd.DataFrame({
            'Feature 1': feature1,
            'Feature 2': feature2,
            'Class': labels,
        })

        st.dataframe(data.style.set_properties(
            caption="Dataset Preview",
            align="center",
            index_label="#",
        ))

        df = pd.DataFrame(data)

        fig, ax = plt.subplots()
        # Create the horizontal barplot
        sns.countplot(y='Class', data=df, hue='Class', palette='bright', ax=ax)

        # Add the title
        ax.set_title('Plot of Target Class Distribution')
        # Display the plot using Streamlit
        st.pyplot(fig)

        if displaysummary:
            # Display other informative elements
            st.header("Data Information")
            st.write(df.describe())  # Include data summary
        if enabledownload:
            # Add download button with enhanced error handling and feedback
            csv_file = BytesIO()
            data.to_csv(csv_file, index=False)
            csv_file.seek(0)

            download_button = st.download_button(
                label="Download CSV",
                data=csv_file,
                file_name="dataset.csv",
                mime="text/csv",
                on_click=None,  # Disable immediate download on page load
            )

            if download_button:
                try:
                    st.success("Download successful!")
                except Exception as e:
                    st.error(f"Download failed: {e}")
                st.write("You can now explore and analyze this dataset for various purposes.")
    
#run the app
if __name__ == "__main__":
    app()
