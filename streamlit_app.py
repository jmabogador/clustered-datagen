#Input the relevant libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.datasets import make_blobs

def generate_random_points_in_square(x_min, x_max, y_min, y_max, num_points):
    """
    Generates a NumPy array of random points within a specified square region.

    Args:
        x_min (float): Minimum x-coordinate of the square.
        x_max (float): Maximum x-coordinate of the square.
        y_min (float): Minimum y-coordinate of the square.
        y_max (float): Maximum y-coordinate of the square.
        num_points (int): Number of points to generate.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (num_points, 2) containing the generated points.
    """

    # Generate random points within the defined square region
    points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(num_points, 2))

    return points

# Define the Streamlit app
def app():
    # Display the DataFrame with formatting
    st.title("Generate Clustered Data")
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

    # Get user's inputs
    n_samples = int(st.number_input("Enter the number of samples:"))
    cluster_std = st.number_input("Standard deviation (between 0 and 1):")
    random_state = st.number_input("Random seed (between 0 and 100):")

    if st.button('Start'):

        centers = generate_random_points_in_square(-4, 4, -4, 4, 4)

        X, y = make_blobs(n_samples=n_samples, n_features=2,
                        cluster_std=cluster_std, centers = centers,
                        random_state=random_state)

        #use the Numpy array to merge the data and test columns
        dataset = np.column_stack((X, y))

        df = pd.DataFrame(dataset)
        # Add column names to the DataFrame
        df = df.rename(columns={0: 'X', 1: 'Y', 2: 'Class'})

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
