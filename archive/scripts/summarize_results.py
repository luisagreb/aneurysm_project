import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/luisa/aneurysm_project/classification_results/Healthy_TAV_BAV/pairwise_comparisons/all_pairwise_comparisons.csv')
sig = df[df['Significant'] == True]

counts = sig.groupby('Comparison').size().to_dict()

# Let's verify the EXACT mapping the user wanted in the screenshot.
# The user's screenshot had:
# Disease effect (No Collagen)
# Disease effect (+Collagen)
# Collagen effect (Healthy)
# Collagen effect (TAA)
# Healthy NoColl vs TAA +Coll

# Mapping the new 6-groups to this type of plot exactly:
map_to_plot = [
    ('Disease effect\n(No Collagen) TAV', 'Disease Effect (No Collagen) TAV-TAA'),
    ('Disease effect\n(+Collagen) TAV', 'Disease Effect (+Collagen) TAV-TAA'),
    ('Disease effect\n(No Collagen) BAV', 'Disease Effect (No Collagen) BAV-TAA'),
    ('Disease effect\n(+Collagen) BAV', 'Disease Effect (+Collagen) BAV-TAA'),
    ('Collagen effect\n(Healthy)', 'Collagen Effect (Healthy)'),
    ('Collagen effect\n(TAV)', 'Collagen Effect (TAV_TAA)'),
    ('Collagen effect\n(BAV)', 'Collagen Effect (BAV_TAA)'),
    ('Healthy NoColl vs\nTAV +Coll', 'Interaction Healthy_NoColl vs TAV_TAA_Coll'), # Wait, I don't have this explicitly in my comparisons list right now! Let's check!
]
