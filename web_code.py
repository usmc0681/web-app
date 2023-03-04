import pandas as pd
import streamlit as st
import xgboost

st.write("# 3 Year Recidivism Risk")

col1, col2, col3, col4, col5 = st.columns(5)

gender = col1.selectbox("Enter your gender",["Male", "Female"])
race = col2.selectbox("Enter race", ["White", "Black"])
age = col3.selectbox("Enter age at release", ["18-22", "23-27", "28-32", "33-37", "38-42", "42-47", "48 or older"])
gang = col4.selectbox("Gang Affiliation", ["TRUE", "FALSE"])
help = col5.button("Help")
sup_score = col1.number_input("Enter Sup Score (1-10)")
sup_level = col2.selectbox("Enter Supervision  Level", ["Standard", "Specialized", "High"])
ed_level = col3.selectbox("Enter Education Level", ["Less than HS diploma", "High School Diploma", "At least some college"])
deps = col4.selectbox("Number of Dependents", ["0", "1", "2", "3 or more"])
prison_off = col1.selectbox("Prison Offense", ["Property", "Drugs", "Violent/Sex", "Violent/Non-Sex", "Other"])
prison_yrs = col2.selectbox("Prison years", ["Less than 1 year", "1-2 years", "Greater than 2 to 3 years", "more than 3 years"])
prior_arr_felony = col3.selectbox("Prior Felony Arrests", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10 or more"])
prior_arr_misd = col4.selectbox("Prior Misd Arrests", ["0", "1", "2", "3", "4", "5", "6 or more"])
prior_arr_vio = col1.selectbox("Prior Violent Arrests", ["0", "1", "2", "3 or more"])
prior_arr_prop = col2.selectbox("Prior Property Arrests", ["0", "1", "2", "3", "4", "5 or more"])
prior_arr_drug = col3.selectbox("Prior Drug Arrests", ["0", "1", "2", "3", "4", "5 or more"])
prior_arr_par_viol = col4.selectbox("Prior Violation Arrests", ["0", "1", "2", "3", "4", "5 or more"])
prior_arr_dv_charge = col1.selectbox("Prior DV Arrests", ["TRUE", "FALSE"])
prior_arr_gun = col2.selectbox("Prior Gun Arrests", ["TRUE", "FALSE"])
prior_conv_felony = col3.selectbox("Prior Convictions Felony", ["0", "1", "2", "3 or more"])
prior_conv_misd = col4.selectbox("Prior Conv Misd", ["0", "1", "2", "3", "4 or more"])
prior_conv_vio = col1.selectbox("Prior Conv Vio", ["TRUE", "FALSE"])
prior_conv_prop = col2.selectbox("Prior Conv Prop", ["1", "2", "3 or more"])
prior_conv_drug = col3.selectbox("Prior Conv Drug", ["0", "1", "2 or more"])
prior_conv_par_viol = col4.selectbox("Prior Conv Parole Vio", ["TRUE", "FALSE"])
prior_conv_dv = col1.selectbox("Prior Conv DV", ["TRUE", "FALSE"])
prior_conv_gun = col2.selectbox("Prior Conv Gun", ["TRUE", "FALSE"])
prior_parole_revo = col3.selectbox("Prior Parole Revocations", ["TRUE", "FALSE"])
prior_prob_revo = col4.selectbox("Prior Probation Revocations", ["TRUE", "FALSE"])
mh_cond = col1.selectbox("Mental Health Case", ["TRUE", "FALSE"])
cog_edu = col2.selectbox("Cognitive ED", ["TRUE", "FALSE"])
other_cond = col3.selectbox("Other Conditions", ["TRUE", "FALSE"])
ele_mon_vio = col4.selectbox("Electronic Monitoring Violation", ["TRUE", "FALSE"])
inst_vio = col1.selectbox("Instruction Violation", ["TRUE", "FALSE"])
fail_to_rpt = col2.selectbox("Failure to report", ["TRUE", "FALSE"])
move_wo_permis = col3.selectbox("Move without permission", ["TRUE", "FALSE"])
del_rpts = col4.selectbox("Delinquency Reports", ["0", "1", "2", "3", "4 or more"])
prog_att =col1.selectbox("Program attendance", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10 or more"])
unex_abs = col2.selectbox("Unexcused absences", ["0", "1", "2", "3 or more"])
resid_chng = col3.selectbox("Residence changes", ["0", "1", "2", "3 or more"])
avg_drug_days = col4.number_input("Average days per drug test")
pos_thc = col1.number_input("Positive THC test")
pos_coc = col2.number_input("Positive Cocaine test")
pos_meth = col3.number_input("Positive Meth test")
pos_other = col4.number_input("Other positive test")
pct_emply = col1.number_input("Percent days employed")
job_per_year = col2.number_input("Jobs per year")
emp_exmpt = col3.selectbox("Employment Exempt", ["TRUE", "FALSE"])

if st.button("Predict"):

    if(prediction[0]==0):
        st.write('<p class = "big-font">3 year Recidivism Risk is low.</p>', unsafe_allow_html= True)

    else:
        st.write('<p class = "big-font">3 year Recidivism Risk is high.</p>', unsafe_allow_html=True)

df_pred = pd.DataFrame([[gender, race, age, gang, sup_score, sup_level, ed_level, deps, prison_off, prison_yrs,
                         prior_arr_felony, prior_arr_misd, prior_arr_vio, prior_arr_prop, prior_arr_drug,
                         prior_arr_par_viol, prior_arr_dv_charge, prior_arr_gun, prior_conv_felony, prior_conv_misd,
                         prior_conv_vio, prior_conv_prop, prior_conv_drug, prior_conv_par_viol, prior_conv_dv,
                         prior_conv_gun, prior_parole_revo, prior_prob_revo, mh_cond, cog_edu, other_cond,
                         ele_mon_vio, inst_vio, fail_to_rpt, move_wo_permis, del_rpts, prog_att, unex_abs,
                         resid_chng, avg_drug_days, pos_thc, pos_coc, pos_meth, pos_other, pct_emply,
                         job_per_year, emp_exmpt]],
columns = ['Gender', 'Race', 'Age_at_Release', 'Gang_Affiliated', 'Supervision_Risk_Score_First', 'Supervision_Level_First',
           'Education_Level', 'Dependents', 'Prison_Offense', 'Prison_Years', 'Prior_Arrest_Episodes_Felony',
           'Prior_Arrest_Episodes_Misd', 'Prior_Arrest_Episodes_Violent', 'Prior_Arrest_Episodes_Property', 'Prior_Arrest_Episodes_Drug',
           'Prior_Arrest_Episodes_PPViolationCharges', 'Prior_Arrest_Episodes_DVCharges', 'Prior_Arrest_Episodes_GunCharges',
           'Prior_Conviction_Episodes_Felony', 'Prior_Conviction_Episodes_Misd', 'Prior_Conviction_Episodes_Viol',
           'Prior_Conviction_Episodes_Prop', 'Prior_Conviction_Episodes_Drug', 'Prior_Conviction_Episodes_PPViolationCharges',
           'Prior_Conviction_Episodes_DomesticViolenceCharges', 'Prior_Conviction_Episodes_GunCharges',
           'Prior_Revocations_Parole', 'Prior_Revocations_Probation', 'Condition_MH_SA', 'Condition_Cog_Ed', 'Condition_Other',
           'Violations_ElectronicMonitoring', 'Violations_Instructions', 'Violations_FailToReport', 'Violations_MoveWithoutPermission',
           'Delinquency_Reports', 'Program_Attendances', 'Program_UnexcusedAbsences', 'Residence_Changes', 'Avg_Days_per_DrugTest',
           'DrugTests_THC_Positive', 'DrugTests_Cocaine_Positive', 'DrugTests_Meth_Positive', 'DrugTests_Other_Positive',
           'Percent_Days_Employed', 'Jobs_Per_Year', 'Employment_Exempt'])

df_pred['Gender'] = df_pred['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

df_pred['Race'] = df_pred['Race'].apply(lambda x: 1 if x == 'White' else 0)

def age_trans(data):
    result = 7
    if(data=='18-22'):
        result = 0
    elif(data=='23-27'):
        result = 1
    elif(data=='28-32'):
        result = 2
    elif(data=='33-37'):
        result = 3
    elif(data=='38-42'):
        result = 4
    elif(data=='43-47'):
        result = 5
    elif(data=='48 or older'):
        result = 6
    return result

df_pred['Age_at_Release'] = df_pred['Age_at_Release'].apply(age_trans)

df_pred['Gang_Affiliated'] = df_pred['Gang_Affiliated'].apply(lambda  x: 1 if x == 'TRUE' else 0)

def sup_level(data):
    result = 3
    if(data == "High"):
        result = 0
    elif(data == 'Standard'):
        result = 1
    elif(data == 'Specialized'):
        result = 2

df_pred['Supervision_Level_First'] = df_pred['Supervision_Level_First'].apply(sup_level)

def ed_level(data):
    result = 3
    if(data=="Less than HS diploma"):
        result = 0
    elif(data=='High School Diploma'):
        result = 1
    elif(data=='Some College'):
        result = 2

df_pred['Education_Level'] = df_pred['Education_Level'].apply(ed_level)

def depend(data):
    result = 4
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data=='3 or more'):
        result = 3

df_pred['Dependents'] = df_pred['Dependents'].apply(depend)

def offense(data):
    result = 5
    if(data=="Other"):
        result = 0
    elif(data=='Property'):
        result = 1
    elif(data=='Drug'):
        result = 2
    elif(data=='Violent/Non-Sex'):
        result = 3
    elif(data=='Violent/Sex'):
        result = 4

df_pred['Prison_Offense'] = df_pred['Prison_Offense'].apply(offense)

def years(data):
    result = 4
    if(data=="Less than 1 year"):
        result = 0
    elif(data=='1-2 years'):
        result = 1
    elif(data=='Greater than 2 to 3 years'):
        result = 2
    elif(data=='More than 3 years'):
        result = 3

df_pred['Prison_Years'] = df_pred['Prison_Years'].apply(years)

def prior_felony_arr(data):
    result = 11
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif (data == '3'):
        result = 3
    elif(data == '4'):
        result = 4
    elif(data == '5'):
        result = 5
    elif(data == '6'):
        result = 6
    elif(data == '7'):
        result = 7
    elif(data == '8'):
        result = 8
    elif(data == '9'):
        result = 9
    elif(data == '10 or more'):
        result = 10
df_pred['Prior_Arrest_Episodes_Felony'] = df_pred['Prior_Arrest_Episodes_Felony'].apply(prior_felony_arr)

def prior_misd_arr(data):
    result = 7
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3'):
        result = 3
    elif(data == '4'):
        result = 4
    elif(data == '5'):
        result = 5
    elif(data == '6 or more'):
        result = 6

df_pred['Prior_Arrest_Episodes_Misd'] = df_pred['Prior_Arrest_Episodes_Misd'].apply(prior_misd_arr)

def prior_vio_arr(data):
    result = 4
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3 or more'):
        result = 3

df_pred['Prior_Arrest_Episodes_Violent'] = df_pred['Prior_Arrest_Episodes_Violent'].apply(prior_vio_arr)

def prior_prop_arr(data):
    result = 7
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif (data == '3'):
        result = 3
    elif(data == '4'):
        result = 4
    elif(data == '5'):
        result = 5
    elif(data == '6 or more'):
        result = 6

df_pred['Prior_Arrest_Episodes_Misd'] = df_pred['Prior_Arrest_Episodes_Misd'].apply(prior_prop_arr)

def prior_drug_arr(data):
    result = 6
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3'):
        result = 3
    elif(data == '4'):
        result = 4
    elif(data == '5 or more'):
        result = 5

df_pred['Prior_Arrest_Episodes_Drug'] = df_pred['Prior_Arrest_Episodes_Drug'].apply(prior_drug_arr)

def prior_ppv_arr(data):
    result = 6
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3'):
        result = 3
    elif(data == '4'):
        result = 4
    elif(data == '5 or more'):
        result = 5

df_pred['Prior_Arrest_Episodes_PPViolationCharges'] = df_pred['Prior_Arrest_Episodes_PPViolationCharges'].apply(prior_ppv_arr)

df_pred['Prior_Arrest_Episodes_DVCharges'] = df_pred['Prior_Arrest_Episodes_DVCharges'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Prior_Arrest_Episodes_GunCharges'] = df_pred['Prior_Arrest_Episodes_GunCharges'].apply(lambda  x: 1 if x == 'TRUE' else 0)

def prior_felony_conv(data):
    result = 4
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3 or more'):
        result = 3

df_pred['Prior_Conviction_Episodes_Felony'] = df_pred['Prior_Conviction_Episodes_Felony'].apply(prior_felony_conv)

def prior_misd_conv(data):
    result = 5
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3'):
        result = 3
    elif(data=='4 or more'):
        result = 4

df_pred['Prior_Conviction_Episodes_Misd'] = df_pred['Prior_Conviction_Episodes_Misd'].apply(prior_misd_conv)

df_pred['Prior_Conviction_Episodes_Viol'] = df_pred['Prior_Conviction_Episodes_Viol'].apply(lambda  x: 1 if x == 'TRUE' else 0)

def prior_prop_conv(data):
    result = 4
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3 or more'):
        result = 3

df_pred['Prior_Conviction_Episodes_Prop'] = df_pred['Prior_Conviction_Episodes_Prop'].apply(prior_prop_conv)

def prior_drug_conv(data):
    result = 3
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2 or more'):
        result = 2

df_pred['Prior_Conviction_Episodes_Drug'] = df_pred['Prior_Conviction_Episodes_Drug'].apply(prior_drug_conv)

df_pred['Prior_Conviction_Episodes_PPViolationCharges'] = df_pred['Prior_Conviction_Episodes_PPViolationCharges'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Prior_Conviction_Episodes_DomesticViolenceCharges'] = df_pred['Prior_Conviction_Episodes_DomesticViolenceCharges'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Prior_Conviction_Episodes_GunCharges'] = df_pred['Prior_Conviction_Episodes_GunCharges'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Prior_Revocations_Parole'] = df_pred['Prior_Revocations_Parole'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Prior_Revocations_Probation'] = df_pred['Prior_Revocations_Probation'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Condition_MH_SA'] = df_pred['Condition_MH_SA'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Condition_Cog_Ed'] = df_pred['Condition_Cog_Ed'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Condition_Other'] = df_pred['Condition_Other'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Violations_ElectronicMonitoring'] = df_pred['Violations_ElectronicMonitoring'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Violations_Instructions'] = df_pred['Violations_Instructions'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Violations_FailToReport'] = df_pred['Violations_FailToReport'].apply(lambda  x: 1 if x == 'TRUE' else 0)

df_pred['Violations_MoveWithoutPermission'] = df_pred['Violations_MoveWithoutPermission'].apply(lambda  x: 1 if x == 'TRUE' else 0)

def del_rpts(data):
    result = 5
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3'):
        result = 3
    elif(data=='4 or more'):
        result = 4

df_pred['Delinquency_Reports'] = df_pred['Delinquency_Reports'].apply(del_rpts)

def prog_att(data):
    result = 11
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif (data == '3'):
        result = 3
    elif(data == '4'):
        result = 4
    elif(data == '5'):
        result = 5
    elif(data == '6'):
        result = 6
    elif(data == '7'):
        result = 7
    elif(data == '8'):
        result = 8
    elif(data == '9'):
        result = 9
    elif(data == '10 or more'):
        result = 10
df_pred['Program_Attendances'] = df_pred['Program_Attendances'].apply(prog_att)

def prog_unex(data):
    result = 4
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3 or more'):
        result = 3

df_pred['Program_UnexcusedAbsences'] = df_pred['Program_UnexcusedAbsences'].apply(prog_unex)

def res_change(data):
    result = 4
    if(data=="0"):
        result = 0
    elif(data=='1'):
        result = 1
    elif(data=='2'):
        result = 2
    elif(data == '3 or more'):
        result = 3

df_pred['Residence_Changes'] = df_pred['Residence_Changes'].apply(res_change)

df_pred['Employment_Exempt'] = df_pred['Employment_Exempt'].apply(lambda  x: 1 if x == 'TRUE' else 0)

model = xgboost.Booster()
model.load_model('recidivism.json')
prediction = model.predict(model)