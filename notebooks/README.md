

0.01 - Helps leep work in chronological order. The structure is PHASE.NOTEBOOK. 
NOTEBOOK is just the Nth notebook in that phase to be created. For phases of the project, we generally use a scheme like the following, but you are welcome to design your own conventions:

0 - Data exploration - often just for exploratory work
1 - Data cleaning and feature creation - often writes data to data/processed or data/interim
2 - Visualizations - often writes publication-ready viz to reports
3 - Modeling - training machine learning models
4 - Publication - Notebooks that get turned directly into reports
pjb - Your initials; this is helpful for knowing who created the notebook and prevents collisions from people working in the same notebook.
data-source-1 - A description of what the notebook covers

Example of structure for notebook -  0.01-pjb-data-source-1.ipynb