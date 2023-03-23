#!/usr/bin/env python
# coding: utf-8

# This is script enables visualization of the effects Conservation Practices (NRS) have on Nitrogen Load.
# Acknowledgement: The NRS document refers to Nitrate-N concentration loss. Our current model calculates Nitrogen Surplus instead of Nitrogen Concentration. Currently, we assume that the change in % Nitrate-N is the same change for % Nitrogen Surplus. 
# 
# 
# ![image-2.png](attachment:image-2.png)

# ### 0. Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
sns.set()
import re

import textwrap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import IPython.display
from IPython.display import display, clear_output
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual


# ### 1. Get data

# In[2]:


df_ifew = pd.read_csv('NRS_IFEWs.csv')
df_pract = pd.read_csv('NRS_py_conservation.csv')


# In[3]:


df_pract


# ### 2. Organize data format

# In[4]:


# No need of silage values
df_ifew = df_ifew.drop(['CornSilageAcresHarvested','CornSilageYield_tonspacre'], axis = 1)

# Deal with Nan values
df_ifew.isna().sum()


# In[5]:


# Calculate based on other around year values
df_ifew = df_ifew.fillna(method = "bfill")

# Deal with Nan values
df_ifew.isna().sum()


# In[6]:


df_pract = df_pract.replace(to_replace=r'^-$', value=0, regex=True)

for i in range(len(df_pract)):
    df_pract['Names List'][i] = " ".join(df_pract['Names List'][i].split())
    
df_pract    


# In[7]:


# convert the 'Date' column to datetime format
df_ifew['Year']= pd.to_datetime(df_ifew['Year'])
df_ifew['Year'] = df_ifew['Year'].dt.year


# In[8]:


# Create dropdown lists
y_lst = (df_ifew['Year'].unique()).tolist() #year
r_lst = ((df_ifew['CountyName'].unique()).tolist()) # region
r_lst.sort() # alphabeticallly
r_lst.insert(0,'Iowa')
p_lst = (df_pract['Names List']).tolist() #practice
#len(r_lst)


# In[9]:


df_r = df_ifew.loc[df_ifew['Year'] == 2015]
nr_change = df_pract.loc[df_pract['Names List']=='Moving from Fall to Spring Pre-plant Application','% AVG Nitrate-N Reduction+'].item()
df_r
Ns = df_r['NitrogenSurplus_kg_ha'].sum()
Nnew = Ns-((int(nr_change)/100)*Ns)


# In[10]:


# Create a function takes the info on the dropdown lists (chosen region) and calculates the Nitrogen surplus based on the chosen practice.

def n_ifew(df_ifew, selectedRegion, practice, df_pract, perc):        
    # Region Selection
    r = selectedRegion
    df = df_ifew
    if (r == 'Iowa'):
        df_r = df.groupby(['Year']).sum()
        df_r = df_r.reset_index()
    else:
        df_r = df[df['CountyName'] == r].copy()
       
    # WHAT IF ACCOUNT FOR MULTIPLE PRACTICES?
    nr_change = df_pract.loc[df_pract['Names List']==practice,'% AVG Nitrate-N Reduction+'].item()
    cy_change = df_pract.loc[df_pract['Names List']==practice,'% Corn Yield Change++']  
    
    
    #calculate values produced by query
    df_r.loc[:,'Nnew'] = df_r['NitrogenSurplus_kg_ha']-((int(nr_change)/100)*df_r['NitrogenSurplus_kg_ha'])
    df_r.loc[:,'Cnew'] = df_r['CornGrainYield_bupacre']+((int(cy_change)/100)*df_r['CornGrainYield_bupacre'])

    # total area in hectares of application of nitrogen - do we consider Soybean crops?
    #df_r['total_area_ha'] = (df_r['CornAcresPlanted']+ df_r['SoybeansAcresPlanted'])/2.57
    # total area in hectares of application of nitrogen - do we not consider Soybean crops?
    df_r.loc[:,'total_area_ha'] = (df_r['CornAcresPlanted'])/2.57

    # Nitrogen Load Old in metric tons
    df_r.loc[:,'Old_N_tons'] = df_r['total_area_ha']*df_r['NitrogenSurplus_kg_ha']/1000
    # Nitrogen Load New in metric tons
    df_r.loc[:,'New_N_tons'] = df_r['total_area_ha']*df_r['Nnew']/1000
    # Corn yield Old in bushels
    df_r.loc[:,'Old_Corn_bushels'] = df_r['CornAcresPlanted']*df_r['CornGrainYield_bupacre']
    # Corn Yield New
    df_r.loc[:,'New_Corn_bushels'] = df_r['CornAcresPlanted']*df_r['Cnew']   

    return df_r


# In[11]:


#%%capture --no-display #hide warning here
trial = (n_ifew(df_ifew, 'Monroe', 'Moving from Fall to Spring Pre-plant Application', df_pract, 50))
trial


# In[12]:


# Plot Nitrogen
# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=trial["Year"], y=trial['Old_N_tons'],
                    mode='lines',
                    name='Historical Nitrogen Load (tons)'))
fig.add_trace(go.Scatter(x=trial["Year"], y=trial['New_N_tons'],
                    mode='lines',
                    name='Possible Nitrogen Load (tons)'))


fig.show()


# In[13]:


# Plot Corn
# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=trial["Year"], y=trial['Old_Corn_bushels'],
                    mode='lines',
                    name='Historical Harvested Corn (bushels)'))
fig.add_trace(go.Scatter(x=trial["Year"], y=trial['New_Corn_bushels'],
                    mode='lines',
                    name='Possible Harvested Corn (bushels)'))


fig.show()


# In[14]:


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(go.Scatter(x=trial["Year"], y=trial['Old_N_tons'],
                    mode='lines',
                    name='Historical Nitrogen Load (tons)'),
             secondary_y=False,
             )
fig.add_trace(go.Scatter(x=trial["Year"], y=trial['New_N_tons'],
                    mode='lines',
                    name='Possible Nitrogen Load (tons)'),
             secondary_y=False,
             )


fig.add_trace(go.Scatter(x=trial["Year"], y=trial['Old_Corn_bushels'],
                    mode='lines',
                    name='Historical Harvested Corn (bushels)'),
             secondary_y=True,
             )
fig.add_trace(go.Scatter(x=trial["Year"], y=trial['New_Corn_bushels'],
                    mode='lines',
                    name='Possible Harvested Corn (bushels)'),
             secondary_y=True,
             )


# Add figure title
fig.update_layout(
    title_text="Change from Conservation Practice"
)

# Set x-axis title
fig.update_xaxes(title_text="Year")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Tons of Nitrogen</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Corn Bushels</b>", secondary_y=True)

fig.show()


# ## Define Sliders Widget

# ## Text Widget

# In[15]:


CP = widgets.Dropdown(
    options = p_lst,
    value='Moving from Fall to Spring Pre-plant Application',
    description='Chosen Conservation Practice',
    disabled=False,
)

CR = widgets.Dropdown(
    options = r_lst,
    value='Monroe',
    description='Observed Region',
    disabled=False,
)


# In[16]:


def f(CP, CR):
    # Region Selection
    r = CR
    df = df_ifew
    if (r == 'Iowa'):
        df_r = df.groupby(['Year']).sum()
        df_r = df_r.reset_index()
    else:
        df_r = df[df['CountyName'] == r].copy()
        
    # WHAT IF ACCOUNT FOR MULTIPLE PRACTICES?
    nr_change = df_pract.loc[df_pract['Names List']==CP,'% AVG Nitrate-N Reduction+'].item()
    cy_change = df_pract.loc[df_pract['Names List']==CP,'% Corn Yield Change++']  
    
    
   #calculate values produced by query
    df_r.loc[:,'Nnew'] = df_r['NitrogenSurplus_kg_ha']-((int(nr_change)/100)*df_r['NitrogenSurplus_kg_ha'])
    df_r.loc[:,'Cnew'] = df_r['CornGrainYield_bupacre']+((int(cy_change)/100)*df_r['CornGrainYield_bupacre'])

    # total area in hectares of application of nitrogen - do we consider Soybean crops?
    #df_r['total_area_ha'] = (df_r['CornAcresPlanted']+ df_r['SoybeansAcresPlanted'])/2.57
    # total area in hectares of application of nitrogen - do we not consider Soybean crops?
    df_r.loc[:,'total_area_ha'] = (df_r['CornAcresPlanted'])/2.57

    # Nitrogen Load Old in metric tons
    df_r.loc[:,'Old_N_tons'] = df_r['total_area_ha']*df_r['NitrogenSurplus_kg_ha']/1000
    # Nitrogen Load New in metric tons
    df_r.loc[:,'New_N_tons'] = df_r['total_area_ha']*df_r['Nnew']/1000
    # Corn yield Old in bushels
    df_r.loc[:,'Old_Corn_bushels'] = df_r['CornAcresPlanted']*df_r['CornGrainYield_bupacre']
    # Corn Yield New
    df_r.loc[:,'New_Corn_bushels'] = df_r['CornAcresPlanted']*df_r['Cnew'] 
    
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(go.Scatter(x=df_r["Year"], y=df_r['Old_N_tons'],
                        mode='lines',
                        name='Historical Nitrogen Load (tons)'),
                 secondary_y=False,
                 )
    fig.add_trace(go.Scatter(x=df_r["Year"], y=df_r['New_N_tons'],
                        mode='lines',
                        name='Possible Nitrogen Load (tons)'),
                 secondary_y=False,
                 )


    fig.add_trace(go.Scatter(x=df_r["Year"], y=df_r['Old_Corn_bushels'],
                        mode='lines',
                        name='Historical Harvested Corn (bushels)'),
                 secondary_y=True,
                 )
    fig.add_trace(go.Scatter(x=df_r["Year"], y=df_r['New_Corn_bushels'],
                        mode='lines',
                        name='Possible Harvested Corn (bushels)'),
                 secondary_y=True,
                 )

    # Add figure title
    fig.update_layout(
        title_text="Change from Conservation Practice"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Year")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Tons of Nitrogen</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Corn Bushels</b>", secondary_y=True)

    fig.show()
    
ui = widgets.HBox([CP, CR])    

out = widgets.interactive_output(f, {'CP': CP, 'CR': CR})

display(ui, out)

