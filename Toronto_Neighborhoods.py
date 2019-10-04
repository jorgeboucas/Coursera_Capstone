# ---
# jupyter:
#   jupytext:
#     cell_markers: '{{{,}}}'
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Import libraries

import pandas as pd
import numpy as np
import sys
import requests

# ### Read data

response = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M")

# ### Parse page
#
# The html tag td can be used to identify rows in the table.

# {{{
# separate lines
response=response.text.split("\n")

# table entries have the td tag
response=[ s for s in response if "td" in s ]

# join them all an separate by the end row tag /tr
response="".join(response)
response=response.split("</tr>")

# remove first and last 4 characteres of each row
response=[ s[4:-5].split("</td><td>") for s in response ]

# make a dataframe from the list of rows
df=pd.DataFrame(response,columns=["Postcode","Borough","Neighbourhood"])
# }}}

# ### Clean dataframe
#
# Here we clean the dataframe as instructed in the assignment.

# {{{
# remove all row that are Borough not assigned or None - this last a result of parsing wrong lines with the td tag
df=df[~df["Borough"].astype(str).isin(["Not assigned","None"])]

# reset index
df.reset_index(inplace=True, drop=True)

# remove html tags
def remove_hrefs(x):
    """
    This function removes the hyperlinks from an html a tag in pure text
    """
    if "href=" in str(x):
        x=x.split(">")[1].split("<")[0]
    return x

df["Borough"]=df["Borough"].apply(lambda x: remove_hrefs(x))
df["Neighbourhood"]=df["Neighbourhood"].apply(lambda x: remove_hrefs(x))

# Not assigned neighborhood get the value of "Borough"
df.loc[df["Neighbourhood"]=="Not assigned","Neighbourhood"]=df.loc[df["Neighbourhood"]=="Not assigned","Borough"]

# Aggregate rows from duplicate postcode entries 
df = df.groupby('Postcode',as_index=False).agg(lambda x:', '.join(list(set(x))))

print(df.shape)
# }}}


