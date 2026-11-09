from agent import CorrectionAgent, Incident

agent = CorrectionAgent()
cases = [
    Incident('S1', 'derive_donnees', 0.88, {'psi': 0.34, 'mae': 51}, {'feature': 'temp'}),
    Incident('S2', 'fausse_alerte', 0.86, {'psi': 0.27, 'mae': 39}, {'feature': 'windspeed'}),
    Incident('S3', 'qualite_donnees', 0.93, {'missing_rate': 0.31}, {'feature': 'hum'}),
    Incident('S4', 'regression_version', 0.84, {'mae_v1': 40, 'mae_v2': 49}, {'version': 'v2'}),
]
for case in cases:
    print(case.incident_id, agent.decide(case))
