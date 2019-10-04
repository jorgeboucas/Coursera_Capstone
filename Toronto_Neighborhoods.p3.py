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
df=pd.DataFrame(response,columns=["Postcode","Borough","Neighborhood"])
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
df["Neighborhood"]=df["Neighborhood"].apply(lambda x: remove_hrefs(x))

# Not assigned neighborhood get the value of "Borough"
df.loc[df["Neighborhood"]=="Not assigned","Neighborhood"]=df.loc[df["Neighborhood"]=="Not assigned","Borough"]

# Aggregate rows from duplicate postcode entries 
df = df.groupby(['Postcode'],as_index=False).agg(lambda x:', '.join(list(set(x))))

print(df.shape)
# }}}

# ### Get coordinates for postal codes

# First try using geocoder.

import geocoder

# {{{
# def getCoor(postal_code):
#     # initialize your variable to None
#     lat_lng_coords = None
#     trial=0
#     # loop until you get the coordinates
#     while(lat_lng_coords is None) or (trial < 5) :
#         g = geocoder.google('{}, Toronto, Ontario'.format(postal_code))
#         lat_lng_coords = g.latlng
#         trial=trial+1
#     if lat_lng_coords is None:
#         print("For", postal_code, "it was not possible to retrieve coordinates.")
#         latitude = np.nan
#         longitude = np.nan 
#     else:
#         latitude = lat_lng_coords[0]
#         longitude = lat_lng_coords[1]
#     return str(latitude)+'::'+str(longitude)

# df["coordidantes"]=df["Postcode"].apply(lambda x: getCoor(x)) getCoor(M1B)
# }}}

# As the block above did not work we try just the simples parts.

# {{{
# print("geocoder:", geocoder.google('{}, Toronto, Ontario'.format("M1B")) )
# print("Function:", getCoor("M1B") )
# }}}

# Retrieving data from geocoder did not work. We are therefore using the suplied table.

pcdf=pd.read_csv("https://cocl.us/Geospatial_data")
pcdf.columns=["Postcode","Latitude","Longitude"]

df=pd.merge(df,pcdf,how="left",on=["Postcode"])
df

# #### Create a map of Toronto with neighborhoods superimposed on top.
#
# We take the average latitude and longitude to center the map.

import folium # map rendering library

# {{{
latitude=np.mean(df['Latitude'])
longitude=np.mean(df['Longitude'])

# create map of New York using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df['Latitude'], df['Longitude'], \
                                           df['Borough'], df['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
# }}}

# ### Check the different borough and choose one to work on.

print(df["Borough"].unique())

# We have choosen central Toronto and will now draw a map with the respective neighborhoods.

# {{{
central_toronto=df[df["Borough"]=="Central Toronto"]
latitude=np.mean(central_toronto['Latitude'])
longitude=np.mean(central_toronto['Longitude'])

# create map of New York using latitude and longitude values
map_central = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(central_toronto['Latitude'], central_toronto['Longitude'], \
                                           central_toronto['Borough'], central_toronto['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_central)  
    
map_central
# }}}

# ### Foursquare credentials and API version

# {{{
CLIENT_ID = '******' # your Foursquare ID
CLIENT_SECRET = '******' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
# }}}

# ### Explore Neighborhoods in Toronto

def getNearbyVenues(names, latitudes, longitudes, radius=500, limit=100):
    """
    This functions returns a data frame of the top 100 locations for each neighborhod given it's latitute and longitude.
    """
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            limit)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


central_toronto_venues = getNearbyVenues(names=central_toronto['Neighborhood'],
                                   latitudes=central_toronto['Latitude'],
                                   longitudes=central_toronto['Longitude']
                                  )

central_toronto_venues.head()

print('There are {} uniques categories.'.format(len(central_toronto_venues['Venue Category'].unique())))

# To properly analyse the different categories we will hot encode them.

# {{{
# one hot encoding
central_onehot = pd.get_dummies(central_toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
central_onehot['Neighborhood'] = central_toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [central_onehot.columns[-1]] + list(central_onehot.columns[:-1])
central_onehot = central_onehot[fixed_columns]

central_onehot.head()
# }}}

central_onehot.shape

# For each neighborhood we get the mean number of times of each categorie.

central_grouped = central_onehot.groupby('Neighborhood').mean().reset_index()
central_grouped

central_grouped.shape


# We now create a dataframe with the top 10 categories of venus for each neighborhood.

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# {{{
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = central_grouped['Neighborhood']

for ind in np.arange(central_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(central_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
# }}}

# ### KMeans clustering
#
# This will be based on the top 10 categories of venus for each respective neighborhood.

from sklearn.cluster import KMeans

# {{{
# set number of clusters
kclusters = 5

central_grouped_clustering = central_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(central_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# }}}

# {{{
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

central_merged = central_toronto

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
central_merged = central_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

central_merged.head() # check the last columns!
# }}}

# ### Dispaly the different clusters on a map.

import matplotlib.cm as cm
import matplotlib.colors as colors

# {{{
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(central_merged['Latitude'], central_merged['Longitude'], \
                                  central_merged['Neighborhood'], central_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
# }}}


