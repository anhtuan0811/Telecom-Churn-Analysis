# 👨🏼‍⚕️ Telecom Churn Analysis


# :books: Table of Contents <!-- omit in toc -->

- [:briefcase: Case Study and Requirement](#case-study-and-requirement)
- [:bookmark_tabs: Example Datasets](#bookmark_tabsexample-datasets)
- [🔎 Explore data and test model](#explore-data-and-test-model)
- [📃 What can you practice with this case study?](#-what-can-you-practice-with-this-case-study)

---

# Case Study and Requirement

The Telco customer churn data contains information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3. It indicates which customers have left, stayed, or signed up for their service.
### ❓ Question
Can you predict whether customers will churn based on customer data ?

# :bookmark_tabs:Example Datasets

<details><summary> 👆🏼 Click to expand Dataset information </summary>
- CustomerID: A unique ID that identifies each customer.
- Gender: Indicate if the customer is a male or a female
- SeniorCitizen: Indicate if the customer is a senior citizen: 1, 0
- Partner: Indicate if the customer has a partner: Yes, No
- Dependents: Indicate if the customer has dependents: Yes, No
- Tenure: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
- Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No
- Avg Monthly Long Distance Charges: Indicates the customer’s average long distance charges, calculated to the end of the quarter specified above.
- Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
- Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
- Avg Monthly GB Download: Indicates the customer’s average download volume in gigabytes, calculated to the end of the quarter specified above.
- Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
- Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
- Device Protection Plan: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
- Premium Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
- Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
- Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
- Streaming Music: Indicates if the customer uses their Internet service to stream music from a third party provider: Yes, No. The company does not charge an additional fee for this service.
- Unlimited Data: Indicates if the customer has paid an additional monthly fee to have unlimited data downloads/uploads: Yes, No
- Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
- Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No
- Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
- Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.
- Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.
- Churn: Indicate if the customer churns: Yes, No

 

</details>

<details><summary> 👆🏼 Click to expand Dataset sample rows </summary>

<div align="center">

**Table** 

<div align="center">
First 10 rows

| customerID   | gender   |   SeniorCitizen | Partner   | Dependents   |   tenure | PhoneService   | MultipleLines    | InternetService   | OnlineSecurity   | OnlineBackup   | DeviceProtection   | TechSupport   | StreamingTV   | StreamingMovies   | Contract       | PaperlessBilling   | PaymentMethod             |   MonthlyCharges |   TotalCharges | Churn   |
|:-------------|:---------|----------------:|:----------|:-------------|---------:|:---------------|:-----------------|:------------------|:-----------------|:---------------|:-------------------|:--------------|:--------------|:------------------|:---------------|:-------------------|:--------------------------|-----------------:|---------------:|:--------|
| 7590-VHVEG   | Female   |               0 | Yes       | No           |        1 | No             | No phone service | DSL               | No               | Yes            | No                 | No            | No            | No                | Month-to-month | Yes                | Electronic check          |            29.85 |          29.85 | No      |
| 5575-GNVDE   | Male     |               0 | No        | No           |       34 | Yes            | No               | DSL               | Yes              | No             | Yes                | No            | No            | No                | One year       | No                 | Mailed check              |            56.95 |        1889.5  | No      |
| 3668-QPYBK   | Male     |               0 | No        | No           |        2 | Yes            | No               | DSL               | Yes              | Yes            | No                 | No            | No            | No                | Month-to-month | Yes                | Mailed check              |            53.85 |         108.15 | Yes     |
| 7795-CFOCW   | Male     |               0 | No        | No           |       45 | No             | No phone service | DSL               | Yes              | No             | Yes                | Yes           | No            | No                | One year       | No                 | Bank transfer (automatic) |            42.3  |        1840.75 | No      |
| 9237-HQITU   | Female   |               0 | No        | No           |        2 | Yes            | No               | Fiber optic       | No               | No             | No                 | No            | No            | No                | Month-to-month | Yes                | Electronic check          |            70.7  |         151.65 | Yes     |
| 9305-CDSKC   | Female   |               0 | No        | No           |        8 | Yes            | Yes              | Fiber optic       | No               | No             | Yes                | No            | Yes           | Yes               | Month-to-month | Yes                | Electronic check          |            99.65 |         820.5  | Yes     |
| 1452-KIOVK   | Male     |               0 | No        | Yes          |       22 | Yes            | Yes              | Fiber optic       | No               | Yes            | No                 | No            | Yes           | No                | Month-to-month | Yes                | Credit card (automatic)   |            89.1  |        1949.4  | No      |
| 6713-OKOMC   | Female   |               0 | No        | No           |       10 | No             | No phone service | DSL               | Yes              | No             | No                 | No            | No            | No                | Month-to-month | No                 | Mailed check              |            29.75 |         301.9  | No      |
| 7892-POOKP   | Female   |               0 | Yes       | No           |       28 | Yes            | Yes              | Fiber optic       | No               | No             | Yes                | Yes           | Yes           | Yes               | Month-to-month | Yes                | Electronic check          |           104.8  |        3046.05 | Yes     |
| 6388-TABGU   | Male     |               0 | No        | Yes          |       62 | Yes            | No               | DSL               | Yes              | Yes            | No                 | No            | No            | No                | One year       | No                 | Bank transfer (automatic) |            56.15 |        3487.95 | No      |

</div>
</div>

</details>

---
## 🔎  Explore data and test model

### The Process is following - [Code & Presentation]or [Only Code]



---

# 🧾 What can you practice with this case study?
- Python
 - Pandas, numpy, matplotlib, seaborn, scipy.
 - Data cleaning, checking for null values, and data transformation.
 - Running models, fitting models, and testing models.
 - Defining functions.
 - Using SMOTE-ENN to address data imbalance.

