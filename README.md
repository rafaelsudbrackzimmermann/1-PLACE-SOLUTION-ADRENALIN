# Load Disaggregation Challenge: Energy Use in Buildings - 2024 - [Gold medal - 1st place - Top 1%] 🥇
- **1- Preprocess Data**: [link](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-Adrenalin-Load-Disaggregation-Challenge/blob/main/Submission%201/code/_1_pre_process.py)
- **2- Model**: [link](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-Adrenalin-Load-Disaggregation-Challenge/blob/main/Submission%201/code/_2_model.py)
- **3- Main Training**: [link](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-Adrenalin-Load-Disaggregation-Challenge/blob/main/Submission%201/code/_4_main_train.py)
- [Site](https://adrenalin.energy/Load-Disaggregation-Challenge-Energy-use-in-buildings) | [Slides](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-ADRENALIN/blob/main/Submission%201/Presentation.pptx) | [Report](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-ADRENALIN/blob/main/Submission%201/Report.docx)
- **Keywords**: Unsupervised Learning, Time Series, Disaggregation
![Project Banner](https://raw.githubusercontent.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-Adrenalin-Load-Disaggregation-Challenge/main/Submission%201/Banner2.png)

#### Project Overview
**The Problem:**
Energy management in buildings often lacks detailed insights due to the absence of extensive metering, making it challenging to identify how specific systems like heating and cooling contribute to overall energy consumption.

**The Solution:**
To address this, our team developed a solution using an unsupervised learning approach with the Adjusted STL (Seasonal-Trend Decomposition using LOESS) algorithm. This method improves upon traditional energy disaggregation techniques by adapting to complex and noisy data without needing predefined labels or extensive historical data inputs.

**Technology and Methodology:**
- **Python and Key Libraries:** Our solution was implemented in Python, utilizing libraries such as `pandas` for data manipulation, `numpy` for numerical operations, and `statsmodels` for the robust implementation of the STL decomposition.
- **STL Decomposition:** We chose STL for its flexibility in handling seasonal variations and its compatibility with unsupervised learning frameworks, allowing it to adapt dynamically to the data's inherent patterns.
- **Adjusted STL Algorithm:** This enhanced version of the standard STL method integrates classical decomposition's stability with STL’s adaptability, ensuring accurate energy use breakdowns even when data exhibit volatile seasonal shifts.

**Impact:**
This unsupervised and adaptable approach allows building managers to effectively monitor and control energy usage for various systems without the need for complex and expensive sub-metering infrastructure. By leveraging basic meter data, our model provides a scalable and economically viable solution for energy management across diverse building environments.

#### Results [link](https://codalab.lisn.upsaclay.fr/competitions/19659#results)
**Training Results:**
- **nMAE**: 0.241 (Ranked 2nd)

**Competition Results:**
- **nMAE**: 0.235 (Ranked 1st)
