import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = np.where(bmi > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad.
# If the value of 'cholestorol' or 'gluc' is 1, make the value 0.
# If the value is more than 1, make the value 1.
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)


# Draw Categorical Plot
def draw_cat_plot():

    """
    Solution that works fine, but have some problems with default tests to the task

    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol',
    # 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    categories = df.groupby(["cardio", "active", "alco", "cholesterol", "gluc", "overweight", "smoke", ])\
        .size().rename("counter").reset_index()

    melted = categories.melt(['counter', 'cardio'])

    fig = sns.catplot(x='variable', y='counter', hue='value', col='cardio', kind='bar', ci=None, data=melted)
    """

    # Solution #2
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco',
    # 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    # You will have to rename one of the collumns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().rename('total').reset_index()

    # Draw the catplot with 'sns.catplot()'
    face_grid = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', ci=None, data=df_cat)
    fig = face_grid.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # For this solution to works properly - change matplotlin version in poetry file to 3.2.2

    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= (df['height'].quantile(0.025))) &
        (df['height'] <= (df['height'].quantile(0.975))) &
        (df['weight'] >= (df['weight'].quantile(0.025))) &
        (df['weight'] <= (df['weight'].quantile(0.975)))
    ]

    # Calculate the correlation matrix
    dfc = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(dfc)

    # Set up the matplotlib figure
    fig = plt.figure(figsize=(9, 9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(dfc, annot=True, linewidths=1, mask=mask, vmax=.8, center=0.09, square=True,
                cbar_kws={'shrink': 0.5}, fmt='.1f')

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
