import numpy as np
import pandas as pd

class InformationValue:
    """ Generates a table with information value
    
        Note:  
        
        Less than 0.02  Not useful for prediction
        0.02 to 0.1     Weak predictive Power
        0.1 to 0.3      Medium predictive Power
        0.3 to 0.5      Strong predictive Power
        >0.5    Suspicious Predictive Power
    """
    
    def __init__(self, x, y, missing_value=False):
        """ In
        
        Args:
            x (series, optional): categorical field to evaluate
            y (series, int): target variable required to compute IV.
                    y must be binary with a value of 0 or 1.
            missing_value (boolean): if set as false by default, ignore
                    null values. If true, then include it as one of the
                    values to compute IV.
        
        Returns:
            
        """
        
        if missing_value:
            self.x = x.fillna('missing')
            self.y = y
        else:
            notnull = ~x.isnull()
            self.x = x[notnull]
            self.y = y[notnull]
        
    def weight_of_evidence(self, fraud, not_fraud):
        return np.log(not_fraud/fraud)
    
    def information_value(self, fraud, not_fraud, woe):
        return (not_fraud - fraud) * woe
        
    def create_table(self, sort='IV', ascending=True):
        """ 
        """

        # Compute counts and distribution across combined, fraud and no-fraud
        count = pd.crosstab(self.x, self.y, margins=True)
        dist = pd.crosstab(self.x, self.y, normalize='columns', margins=True)
        
        # Format Table
        count.columns = ['{0}_Count'.format(c) for c in ['No_Fraud','Fraud','All']]
        dist.columns = ['{0}_Dist'.format(c) for c in ['No_Fraud','Fraud','All']]
        table = pd.concat([dist, count], axis=1) 
        table = table.reindex_axis(sorted(table.columns), axis=1) 
        
        # Compute weight-of-evidence (WOE)
        table['WOE'] = self.weight_of_evidence(table.Fraud_Dist, table.No_Fraud_Dist)
    
        # Compute IV across values
        table['IV'] = self.information_value(table.Fraud_Dist, table.No_Fraud_Dist, table.WOE)
        table['IV'] = table.IV.map(lambda x: 0.5 if x >= 0.5 else x)
        
        # Compute total IV
        table.loc['All','IV'] = table['IV'].sum()
        return table.sort_values(by=sort, ascending=ascending)