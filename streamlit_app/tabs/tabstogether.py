import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Introduction import Introduction_content
from tab1 import tab1_content
from tab2 import tab2_content
from tab3 import tab3_content
from tab4 import tab4_content
from tab5 import tab5_content
from tab6 import tab6_content
from conclusion import conclusion


# Define tab names
tabs = {
    "Introduction": Introduction_content,
    "Data Preprocessing": tab1_content,
    'Data Exploration': tab3_content,
    "Data Visualization": tab2_content,
    "Modelling - Classification": tab5_content,
    "Modelling - Regression": tab4_content,
    "Demo": tab6_content,
    "Conclusion": conclusion
}

# Create sidebar for navigation
selected_tab = st.sidebar.radio("Go to", list(tabs.keys()))

# Display selected tab content
tabs[selected_tab]()
