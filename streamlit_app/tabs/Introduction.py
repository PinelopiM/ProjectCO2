import streamlit as st

def Introduction_content():

    title = "Examination of the technical features of cars that influence CO2 emissions"

    class Member:
        def __init__(self, name: str, linkedin_url: str = None, github_url: str = None) -> None:
            self.name = name
            self.linkedin_url = linkedin_url
            self.github_url = github_url

        def sidebar_markdown(self):
            markdown = f'<b style="display: inline-block; vertical-align: middle; height: 100%">{self.name}</b>'

            for platform, url in [("linkedin", self.linkedin_url), ("github", self.github_url)]:
                if url:
                    markdown += f' <a href={url} target="_blank"><img src="https://dst-studio-template.s3.eu-west-3.amazonaws.com/{platform}-logo-black.png" alt="{platform}" width="{25 if platform == "linkedin" else 20}" style="vertical-align: middle; margin-left: 5px"/></a> '

            return markdown

    TITLE = title

    TEAM_MEMBERS = [
        Member(name="Christin Erdmann", github_url="https://github.com/ChrisDataScientist"),
        Member(name="Pinelopi Moutesidi", linkedin_url="https://www.linkedin.com/in/pinelopi-moutesidi-474201266/", github_url="https://github.com/PinelopiM"),
        Member(name="Sanjana Singh", github_url="https://github.com/sanasingh21"),
    ]

    PROMOTION = "Promotion Bootcamp Data Scientist - February 2024"

    # Streamlit layout
    st.title(TITLE)

    st.write(f" {PROMOTION}")
             
    for member in TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    st.write("### Team Members:")
    for member in TEAM_MEMBERS:
        st.write(f"- {member.name}")

    st.image("/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/Intro_photo.jpg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    st.write("The project seeks to conduct a comparative analysis of CO2 emissions originating from private vehicles, pinpoint technical characteristics of cars influencing their CO2 emissions, and forecast CO2 emissions from vehicles based on their design. This analysis can potentially assist the selection of more eco-friendly vehicles in the future.")

