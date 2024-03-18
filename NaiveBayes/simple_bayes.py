import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

if __name__ == "__main__":
    rain_model = BayesianNetwork([
        ('LowPressure', 'Rain'),
        ('Rain', 'Traffic')
    ])

    cpd_low_pressure = TabularCPD(
        variable='LowPressure',
        variable_card=2,
        values=[[0.70], [0.30]]
    )

    cpd_rain = TabularCPD(
        variable='Rain',
        variable_card=2,
        values=[
            [0.80, 0.20],
            [0.20, 0.80]
        ],
        evidence=['LowPressure'],
        evidence_card=[2],
    )

    cpd_traffic = TabularCPD(
        variable='Traffic',
        variable_card=2,
        values=[
            [0.70, 0.10],
            [0.30, 0.90]
        ],
        evidence=['Rain'],
        evidence_card=[2]
    )

    rain_model.add_cpds(
        cpd_low_pressure, cpd_rain, cpd_traffic
    )

    print(rain_model.nodes())
    print(rain_model.edges())

    rain_infer = VariableElimination(rain_model)

    #print(rain_infer.query(variables=['Traffic'], evidence={'Rain': 1}))
    print(rain_infer.query(variables=['LowPressure'], evidence={'Traffic': 1, 'Rain': 1}))
