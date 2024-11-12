# Load Disaggregation Challenge: Energy Use in Buildings - 2024 

- **1- Preprocess Data**: [link](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-Adrenalin-Load-Disaggregation-Challenge/blob/main/Submission%201/code/_1_pre_process.py)
- **2- Model**: [link](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-Adrenalin-Load-Disaggregation-Challenge/blob/main/Submission%201/code/_2_model.py)
- **3- Main Training**: [link](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-Adrenalin-Load-Disaggregation-Challenge/blob/main/Submission%201/code/_4_main_train.py)
- [Site](https://adrenalin.energy/Load-Disaggregation-Challenge-Energy-use-in-buildings) | [Slides](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-ADRENALIN/blob/main/Submission%201/Presentation.pptx) | [Report](https://github.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-ADRENALIN/blob/main/Submission%201/Report.docx)
- **Keywords**: Unsupervised Learning, Time Series, Disaggregation
![Project Banner](https://raw.githubusercontent.com/rafaelsudbrackzimmermann/1-PLACE-SOLUTION-Adrenalin-Load-Disaggregation-Challenge/main/Submission%201/Banner2.png)

#### Project Overview

**Background:**
Buildings are a significant consumer of global energy. Reducing energy usage in buildings is crucial for achieving global emissions targets. The process of building energy load disaggregation helps identify the specific services like heating, cooling, and lighting that consume energy within a building. This identification is vital for targeted energy efficiency measures, enabling building owners and operators to reduce consumption and costs.

**Challenge:**
Accurate disaggregation of individual energy services' consumption from main meter data is a complex problem. Most buildings only have main meter data available, making it challenging to disaggregate specific energy uses like heating and cooling without costly and intrusive sub-metering systems.

**Goal:**
The Load Disaggregation Challenge focuses on developing scalable, unsupervised algorithms capable of accurately disaggregating building energy usage, specifically heating and/or cooling, from main meter data. The challenge tackles the issues of complexity in algorithm design, generalization across diverse building types, and the limited resolution of data logging by main meters.

**Data Sets:**
The datasets provided by competition sponsors include main meter and weather data from buildings with comprehensive sub-metering systems. These "ground truth" data are used to evaluate the accuracy of the disaggregation algorithms developed by participants.


#### Results [link](https://codalab.lisn.upsaclay.fr/competitions/19659#results)
**Training Results:**
- **nMAE**: 0.241 (Ranked 2nd)

**Competition Results:**
- **nMAE**: 0.235 (Ranked 1st)
