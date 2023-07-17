# ----------------------------------------------------- #
    # Evidently function and streamlit 
from evidently import ColumnMapping
from evidently.options import ColorOptions
from evidently.report import Report
import streamlit as st


from evidently.metrics import ClassificationQualityMetric
from evidently.metrics import ClassificationClassBalance
from evidently.metrics import ClassificationConfusionMatrix
from evidently.metrics import ClassificationQualityByClass
from evidently.metrics import ClassificationClassSeparationPlot
from evidently.metrics import ClassificationProbDistribution
from evidently.metrics import ClassificationRocCurve
from evidently.metrics import ClassificationPRCurve
from evidently.metrics import ClassificationPRTable
from evidently.metrics import ClassificationQualityByFeatureTable
from evidently.metrics import ClassificationDummyMetric
from evidently.metrics import ConflictTargetMetric
from evidently.metrics import ConflictPredictionMetric
import streamlit.components.v1 as components
from os.path import exists

from evidently.metrics import DataDriftTable

import pandas as pd

def fit_and_save_report(classifier, ref_data, current_data, label):
    
    ref_data_dd = ref_data # because of error
    current_data_dd = current_data # because of error
    
    new_columns = 'target'
    if 'fraud_bool' in ref_data.columns :
        #label == fraud_bool 
        ref_data_nn = pd.DataFrame(ref_data.drop(label, axis=1))
        current_data_nn = pd.DataFrame(current_data.drop(label, axis=1))
        
        # Then rename
        ref_data_dd = ref_data_dd.rename(columns={'fraud_bool' : new_columns})
        current_data_dd = current_data_dd.rename(columns={'fraud_bool' : new_columns})
        
    else:
        label = new_columns
        ref_data_nn = pd.DataFrame(ref_data.drop(label, axis=1))
        current_data_nn = pd.DataFrame(current_data.drop(label, axis=1))
        # label = 'fraud_bool'
            
    ref_data_dd['prediction'] = classifier.predict_proba(ref_data_nn)[:,1]
    current_data_dd['prediction'] = classifier.predict_proba(current_data_nn)[:,1]
    
    # Debugging 
    # st.dataframe(ref_data)
    # st.dataframe(current_data)
    
    #probabilistic binary classification
    classification_report = Report(metrics=[
        DataDriftTable(num_stattest='kl_div', cat_stattest='psi'),    
        # ClassificationQualityMetric(),
        # ClassificationClassBalance(),
        # ConflictTargetMetric(),
        # ConflictPredictionMetric(),
        # ClassificationConfusionMatrix(),
        # ClassificationQualityByClass(),
        # ClassificationDummyMetric(),
        # ClassificationClassSeparationPlot(),
        # ClassificationProbDistribution(),
        # ClassificationRocCurve(),
        # ClassificationPRCurve(),
        # ClassificationPRTable(),
        # ClassificationQualityByFeatureTable(columns=['mean area', 'fractal dimension error']),
    ])

    classification_report.run(reference_data=ref_data_dd, current_data=current_data_dd)
    
    # st.write(classification_report)
    classification_report.save_html('reports/evidently/classification_report.html')
    # return classification_report
    
    report_path = "reports/evidently/classification_report.html"
    width = 2000
    height = 1000
    if  exists(report_path) == True:
        with open(report_path, encoding="utf8") as report_f:
            report: Text = report_f.read()
            components.html(report, width=width, height=height, scrolling=True)

# ----------------------------------------------------- #