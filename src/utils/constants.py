list_order = ['encounter_id','patient_nbr','race','gender','age','weight','admission_type_id','discharge_disposition_id',
              'admission_source_id','time_in_hospital','payer_code','medical_specialty','num_lab_procedures',
              'num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','diag_1','diag_2',
              'diag_3','number_diagnoses','max_glu_serum','A1Cresult','change','diabetesMed','metformin','repaglinide','nateglinide','chlorpropamide',
              'glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol',
              'troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin',
              'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','readmitted']

list_numeric_int_Deabetic = ['encounter_id','patient_nbr','admission_type_id','discharge_disposition_id','admission_source_id',
                             'time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient',
                             'number_emergency','number_inpatient','number_diagnoses']

list_numeric_int_admission_type = ['admission_type_id']

list_numeric_intdischarge_disposition = ['discharge_disposition_id']

list_numeric_int_admission_source = ['admission_source_id']

medicamentos = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide',
                'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin',
                'tolazamide','metformin-pioglitazone', 'metformin-rosiglitazone', 'glipizide-metformin','troglitazone',
                'tolbutamide', 'acetohexamide']

ML_training_features = ['admission_type_id', 'discharge_disposition_id','admission_source_id', 'diag_1',
                        'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult','race', 'gender', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
                        'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                        'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed','target']