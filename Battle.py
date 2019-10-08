# -*- coding: utf-8 -*-
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

import pandas as pd
import numpy as np
import os
import sys
import requests
from area import area
import folium
from folium import plugins
from shapely.geometry import Point, Polygon
import json

# {{{
# https://public.opendatasoft.com/explore/dataset/postleitzahlen-deutschland/export/?refine.note=Köln
# https://public.opendatasoft.com/explore/dataset/postleitzahlen-deutschland/export/?refine.note=Köln
# }}}

# {{{
# #!wget https://public.opendatasoft.com/explore/dataset/postleitzahlen-deutschland/download/?format=csv&refine.note=Köln&timezone=Europe/Berlin&use_labels_for_header=true
# #!wget https://www.offenedaten-koeln.de/sites/default/files/2014-01-01_Straßenverzeichnis_Stadtviertel.csv
# }}}

# #!wget https://www.suche-postleitzahl.org/download_files/public/plz_einwohner.csv
einwdf=pd.read_csv("plz_einwohner.csv")
einwdf.head()

# #!wget https://www.suche-postleitzahl.org/download_files/public/zuordnung_plz_ort.csv
zuordf=pd.read_csv("2014-01-01_Straßenverzeichnis_Stadtviertel.csv")
zuordf.head()

# #!wget -O coordinates.csv https://public.opendatasoft.com/explore/dataset/postleitzahlen-deutschland/download/?format=csv 
coordf=pd.read_csv("coordinates.csv",sep=";")
coordf.head()



kdf=pd.merge(einwdf,zuordf,on=["plz"],how="right")
kdf.head()

kdf=pd.merge(kdf,coordf,on="plz",how="left")
kdf.head()

kdf["latitude"]=kdf["geo_point_2d"].apply(lambda x: float(x.split(",")[0]))
kdf["longitude"]=kdf["geo_point_2d"].apply(lambda x: float(x.split(",")[1]))

kdf["area"]=kdf["geo_shape"].apply(lambda x: area(x))

kdf["einwohner/area"]=kdf["einwohner"]/kdf["area"]

kdf.head()

plzgeo=kdf[["plz","latitude","longitude"]].groupby(["plz"],as_index=False).mean()
plzgeo.head()

# {{{
# # !wget -O koeln.geo.json "https://public.opendatasoft.com/explore/dataset/postleitzahlen-deutschland/download/?format=geojson&refine.note=Köln&timezone=Europe/Berlin"
# }}}

# {{{
# # !wget -O germany.geo.json "https://public.opendatasoft.com/explore/dataset/postleitzahlen-deutschland/download/?format=geojson&timezone=Europe/Berlin"
# }}}

#plzsdf=plzgeo.copy()
plzsdf=kdf[["plz","stadtteilname"]].groupby(["plz"], as_index=False).agg({"stadtteilname": lambda x: ", ".join(list(set(x)))})
plzsdf["stadtteilname"]=plzsdf["plz"]+": "+plzsdf["stadtteilname"]
plzsdf=pd.merge(plzsdf,plzgeo)
plzsdf.head()


latitude=np.mean(kdf["latitude"])
longitude=np.mean(kdf["longitude"])
kdf["plz"]=kdf["plz"].astype(str)
map_values=kdf[["plz","einwohner","einwohner/area","area"]].drop_duplicates()
map_values["einwohner/area"]=map_values["einwohner/area"].apply(lambda x: np.log2(x))

# {{{
koeln_geo = r'koeln.geo.json' # geojson file
germany_geo=r'germany.geo.json'

# create a plain world map
koln_map = folium.Map(location=[latitude, longitude], zoom_start=10.5, tiles='Mapbox Bright')

folium.Choropleth(
    geo_data=koeln_geo,
    data=map_values,
    columns=['plz', 'einwohner'],
    key_on='feature.properties.plz', # 'feature.properties.name'
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Inhabitants'
).add_to(koln_map)


# let's start again with a clean copy of the map of San Francisco
sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# instantiate a mark cluster object for the incidents in the dataframe
#plzs = plugins.MarkerCluster().add_to(koln_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(plzsdf["latitude"], plzsdf["longitude"], plzsdf["stadtteilname"]):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(koln_map) # plzs

# display map
koln_map
# }}}

# {{{
koeln_geo = r'koeln.geo.json' # geojson file
germany_geo=r'germany.geo.json'

# create a plain world map
koln_map = folium.Map(location=[latitude, longitude], zoom_start=10.5, tiles='Mapbox Bright')

folium.Choropleth(
    geo_data=koeln_geo,
    data=map_values,
    columns=['plz', 'einwohner/area'],
    key_on='feature.properties.plz', # 'feature.properties.name'
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='log2(inhabitants/area)'
).add_to(koln_map)


# let's start again with a clean copy of the map of San Francisco
sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# instantiate a mark cluster object for the incidents in the dataframe
#plzs = plugins.MarkerCluster().add_to(koln_map)

# loop through the dataframe and add each data point to the mark cluster
# for lat, lng, label, in zip(plzsdf["latitude"], plzsdf["longitude"], plzsdf["stadtteilname"]):
#     folium.Marker(
#         location=[lat, lng],
#         icon=None,
#         popup=label,
#     ).add_to(koln_map) # plzs

# display map
koln_map
# }}}

# {{{
outcoord=kdf[["geo_shape"]].drop_duplicates()
def getMaxXCoor(coords):
    coords = json.loads(coords)
    coords = coords['coordinates'][0]
    coords_1 = [ s[0] for s in coords ]
    coords_2 = [ s[1] for s in coords ]
    coords_1_max=str(max(coords_1))
    coords_1_min=str(min(coords_1))
    coords_2_max=str(max(coords_2))
    coords_2_min=str(min(coords_2))
    results=[coords_1_max,coords_1_min,coords_2_max,coords_2_min]
    results="; ".join(results)
    return results
    
outcoord["max;min"]=outcoord["geo_shape"].apply(lambda x: getMaxXCoor(x))
outcoord[['coords_1_max','coords_1_min','coords_2_max','coords_2_min']]=outcoord["max;min"].str.split('; ',expand=True)
coords_1_max=max(outcoord["coords_1_max"].tolist())
coords_1_min=min(outcoord["coords_1_min"].tolist())
coords_2_max=max(outcoord["coords_2_max"].tolist())
coords_2_min=min(outcoord["coords_2_min"].tolist())

print(coords_1_max,coords_1_min,coords_2_max,coords_2_min)
# }}}

# {{{
nlat=38

sizelat=float(coords_1_max)-float(coords_1_min)
sizelong=float(coords_2_max)-float(coords_2_min)

addon=sizelat/nlat
new_lats=[float(coords_1_min)+addon/2]
while len( new_lats ) < nlat :
    new_lats.append(new_lats[-1]+addon)
    
nlong=int(sizelong*nlat/sizelat)
print(nlong)

addon=sizelong/nlong
new_longs=[float(coords_2_min)+addon/2]
while len( new_longs ) < nlong :
    new_longs.append(new_longs[-1]+addon)

# }}}

pols=kdf["geo_shape"].unique()

# {{{
import itertools

query_coordinates = list(itertools.product(new_lats,new_longs))
query_coordinates = [ list(s) for s in query_coordinates ]
query_coordinates = list(itertools.product(query_coordinates,pols))
#query_coordinates = [ s[0].append(s[1]) for s in query_coordinates ]
query_coordinates = [ list(s) for s in query_coordinates ]
#query_coordinates = [ s[0].append(s[1]) for s in query_coordinates ]
query_coordinates = [ s[0]+[s[1]] for s in query_coordinates ]
query_coordinates = pd.DataFrame(query_coordinates, columns=['latitude','longitude','polygon'])

def checkquerypoint(df):
    lat=df['latitude']
    lng=df["longitude"]
    polygon=df["polygon"]
    
    p1=Point(lat,lng)
    coords = json.loads(polygon)
    coords = coords['coordinates'][0]
    coords = [ tuple(s) for s in coords ]
    poly=Polygon(coords)
    if p1.within(poly):
        return "yes"
    else:
        return "no"


query_coordinates["koeln"]=query_coordinates.apply(checkquerypoint, axis=1)
query_coordinates=query_coordinates[query_coordinates["koeln"]=="yes"][['latitude','longitude',"polygon"]].drop_duplicates()
query_coordinates["n"]=query_coordinates.index.tolist()
query_coordinates["n"]=query_coordinates["n"].astype(str)
query_coordinates.reset_index(inplace=True, drop=True)
print(len(query_coordinates)) #479
# }}}

# {{{
# query_coordinates=pd.DataFrame(query_coordinates,columns=["latitude","longitude"])
# query_coordinates["n"]=query_coordinates.index.tolist()
# query_coordinates["n"]=query_coordinates["n"].astype(str)
# query_coordinates.head()
# }}}

# {{{
koeln_geo = r'koeln.geo.json' # geojson file
germany_geo=r'germany.geo.json'

# create a plain world map
koln_map = folium.Map(location=[latitude, longitude], zoom_start=10.5, tiles='Mapbox Bright')

folium.Choropleth(
    geo_data=koeln_geo,
    data=map_values,
    columns=['plz', 'einwohner'],
    key_on='feature.properties.plz', # 'feature.properties.name'
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Inhabitants'
).add_to(koln_map)


# let's start again with a clean copy of the map of San Francisco
sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# instantiate a mark cluster object for the incidents in the dataframe
#plzs = plugins.MarkerCluster().add_to(koln_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(query_coordinates["longitude"], query_coordinates["latitude"], query_coordinates["n"]):
    folium.CircleMarker(
        [lat, lng],
        radius=2,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(koln_map) 
    

# display map
koln_map
# }}}
# {{{
from math import sin, cos, sqrt, atan2, radians
R = 6373.0
n1="10193"
n2="10241"
lat1 = radians(query_coordinates[query_coordinates["n"]==n1]["latitude"].tolist()[0])
lon1 = radians(query_coordinates[query_coordinates["n"]==n1]["longitude"].tolist()[0])
lat2 = radians(query_coordinates[query_coordinates["n"]==n2]["latitude"].tolist()[0])
lon2 = radians(query_coordinates[query_coordinates["n"]==n2]["longitude"].tolist()[0])

dlon = lon2 - lon1
dlat = lat2 - lat1

a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
c = 2 * atan2(sqrt(a), sqrt(1 - a))

distance = R * c
print(distance)
query_radius=distance/4*1000.00
print(query_radius)
# }}}

help(round)

# {{{
query_coordinates["latitude"]=query_coordinates["latitude"].astype(float)
query_coordinates["longitude"]=query_coordinates["longitude"].astype(float)

query_coordinates["latitude"]=query_coordinates["latitude"].apply(lambda x: round(x, 5))
query_coordinates["longitude"]=query_coordinates["longitude"].apply(lambda x: round(x, 6))
# }}}

query_coordinates.head()

# {{{
# https://api.foursquare.com/v2/venues/explore?&client_id=BVT1SFWBJO55GRF3FRC5E4ZFYLW0UT3CQHPIMYXAPADQWERL&client_secret=WPPLCOY4SVW1SDCI54VZDEENTZPVZ0DLCXZVQUNYSWT0OGZB&v=20180605&ll=50.941810,6.95190,&radius=300&limit=300
# }}}

# {{{
CLIENT_ID = 'BVT1SFWBJO55GRF3FRC5E4ZFYLW0UT3CQHPIMYXAPADQWERL' # your Foursquare ID
CLIENT_SECRET = 'WPPLCOY4SVW1SDCI54VZDEENTZPVZ0DLCXZVQUNYSWT0OGZB' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

def checkpoint(df):
    lat=df['Venue Latitude']
    lng=df["Venue Longitude"]
    polygon=df["polygon"]
    
    p1=Point(lng,lat)
    coords = json.loads(polygon)
    coords = coords['coordinates'][0]
    coords = [ tuple(s) for s in coords ]
    poly=Polygon(coords)
    if p1.within(poly):
        return "yes"
    else:
        return "no"

def getNearbyVenues(names, latitudes, longitudes, polygon, radius=query_radius, limit=1000):
    
    
    venues_list=[]
    for name, lat, lng, pol in zip(names, latitudes, longitudes, polygon):
        #print(name, lat, lng)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            limit)
        
        #print(url)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        #print(results)
        if len(results) > 0:
            # return only relevant information for each nearby venue
            venues_list.append([(
                name, 
                lat, 
                lng, 
                v['venue']['name'], 
                v['venue']['location']['lat'], 
                v['venue']['location']['lng'],  
                v['venue']['categories'][0]['name'],
                pol) for v in results]
            )
    if len(venues_list) > 0 :
        
        nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
        nearby_venues.columns = ['Neighborhood', 
                      'Neighborhood Latitude', 
                      'Neighborhood Longitude', 
                      'Venue', 
                      'Venue Latitude', 
                      'Venue Longitude', 
                      'Venue Category',
                      'polygon']
        #nearby_venues["polygon"]=polygon

        nearby_venues["in plz"]=nearby_venues.apply(checkpoint,axis=1)
        response_size=len(nearby_venues)
        nearby_venues["n response"]=response_size
        
    else:
        nearby_venues=pd.DataFrame(columns = ['Neighborhood', 
                      'Neighborhood Latitude', 
                      'Neighborhood Longitude', 
                      'Venue', 
                      'Venue Latitude', 
                      'Venue Longitude', 
                      'Venue Category',
                      'polygon',"in plz","n response"])
    
    return(nearby_venues)

# koeln_venues= getNearbyVenues(names=query_coordinates[query_coordinates['n']=="20064"]['n'],
#                                    latitudes=query_coordinates[query_coordinates['n']=="20064"]['longitude'],
#                                    longitudes=query_coordinates[query_coordinates['n']=="20064"]['latitude'],
#                                    polygon=query_coordinates[query_coordinates['n']=="20064"]['polygon'])

koeln_venues= getNearbyVenues(names=query_coordinates['n'],
                                   latitudes=query_coordinates['longitude'],
                                   longitudes=query_coordinates['latitude'],
                                   polygon=query_coordinates['polygon'])

print(len(koeln_venues[koeln_venues["in plz"]=="yes"]))
print(len(koeln_venues[koeln_venues["in plz"]=="no"]))
# }}}

len(koeln_venues["polygon"].unique())


kdf.head()

koeln_venues=koeln_venues.drop(["plz","area"],axis=1)

koeln_venues.head()
# koeln_venues=koeln_venues[koeln_venues["in plz"]=="yes"][["polygon","Venue",\
#                                                           "Venue Latitude","Venue Longitude",\
#                                                           "Venue Category"]]
plz_pol=kdf[["plz","geo_shape","area","einwohner"]].drop_duplicates()
plz_pol=plz_pol.rename(columns={"geo_shape":"polygon"})
koeln_venues=pd.merge(plz_pol,koeln_venues,on=["polygon"],how="inner")
koeln_venues

koeln_venues=koeln_venues.drop_duplicates(subset=["Venue","Venue Latitude","Venue Longitude"])

n_venues=koeln_venues[["plz","Venue"]].groupby(["plz"],as_index=False).count()
n_venues=pd.merge(n_venues,koeln_venues[["plz","area","einwohner"]].drop_duplicates(),on=["plz"],how="inner")
n_venues["venues/area"]=np.log2(n_venues["Venue"]/n_venues["area"])
n_venues["inhabitants/venue"]=np.log2(n_venues["einwohner"]/n_venues["Venue"])

# {{{
koeln_geo = r'koeln.geo.json' # geojson file
germany_geo=r'germany.geo.json'

# create a plain world map
koln_map = folium.Map(location=[latitude, longitude], zoom_start=10.5, tiles='Mapbox Bright')

folium.Choropleth(
    geo_data=koeln_geo,
    data=n_venues,
    columns=['plz', "venues/area"],
    key_on='feature.properties.plz', # 'feature.properties.name'
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='log2(venues/area)'
).add_to(koln_map)


# # let's start again with a clean copy of the map of San Francisco
# sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# # instantiate a mark cluster object for the incidents in the dataframe
# #plzs = plugins.MarkerCluster().add_to(koln_map)

# # loop through the dataframe and add each data point to the mark cluster
# for lat, lng, label, in zip(query_coordinates["longitude"], query_coordinates["latitude"], query_coordinates["n"]):
#     folium.CircleMarker(
#         [lat, lng],
#         radius=2,
#         popup=label,
#         color='blue',
#         fill=True,
#         fill_color='#3186cc',
#         fill_opacity=0.7,
#         parse_html=False).add_to(koln_map) 
    

# display map
koln_map.save("venues.html")
koln_map
# }}}

# {{{
koeln_geo = r'koeln.geo.json' # geojson file
germany_geo=r'germany.geo.json'

# create a plain world map
koln_map = folium.Map(location=[latitude, longitude], zoom_start=10.5, tiles='Mapbox Bright')

folium.Choropleth(
    geo_data=koeln_geo,
    data=n_venues,
    columns=['plz', "inhabitants/venue"],
    key_on='feature.properties.plz', # 'feature.properties.name'
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='inhabitants/venue'
).add_to(koln_map)


# # let's start again with a clean copy of the map of San Francisco
# sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# # instantiate a mark cluster object for the incidents in the dataframe
# #plzs = plugins.MarkerCluster().add_to(koln_map)

# # loop through the dataframe and add each data point to the mark cluster
# for lat, lng, label, in zip(query_coordinates["longitude"], query_coordinates["latitude"], query_coordinates["n"]):
#     folium.CircleMarker(
#         [lat, lng],
#         radius=2,
#         popup=label,
#         color='blue',
#         fill=True,
#         fill_color='#3186cc',
#         fill_opacity=0.7,
#         parse_html=False).add_to(koln_map) 
    

# display map
koln_map.save("venues2.html")
koln_map
# }}}

kdf.head()

koeln_venues.head()



from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt


X=StandardScaler().fit_transform(X)
sse = {}
for k in range(1, 43):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()    

# {{{

# forModel[forModel["Cluster Labels"]==forModel[forModel['plz']=="50733"]["Cluster Labels"].tolist()[0] ]
# }}}

kclusters=4
forModel=koeln_venues[["plz","einwohner","area","Venue Category"]]
forModel=pd.merge(forModel,n_venues[["plz","venues/area"]],on=["plz"],how="outer")
forModel["einwohner"]=np.log2(forModel["einwohner"]/forModel["area"])
forModel=forModel.drop(["area"],axis=1)
forModel.reset_index(inplace=True, drop=True)
forModel_onehot = pd.get_dummies(forModel[['Venue Category']], prefix="", prefix_sep="")
forModel=forModel.drop(["Venue Category"],axis=1)
forModel=pd.concat([forModel[["plz"]],forModel_onehot],axis=1)
# forModel=pd.concat([forModel,forModel_onehot],axis=1)
forModel = forModel.groupby('plz').mean().reset_index()
X = forModel.drop(["plz"],axis=1)
#X.head()
#X=StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(X)
forModel.insert(0, 'Cluster Labels', kmeans.labels_)
koelnclusters=pd.merge( forModel[["plz","Cluster Labels"]], kdf[["plz","latitude","longitude"]].drop_duplicates() )

# {{{
koeln_geo = r'koeln.geo.json' # geojson file
germany_geo=r'germany.geo.json'

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

koln_map = folium.Map(location=[latitude, longitude], zoom_start=10.5, tiles='Mapbox Bright')

folium.Choropleth(
    geo_data=koeln_geo,
    data=n_venues,
    columns=['plz', "venues/area"],
    key_on='feature.properties.plz', # 'feature.properties.name'
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='log2(venues/area)'
).add_to(koln_map)


# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(koelnclusters['latitude'], koelnclusters['longitude'], \
                                  koelnclusters['plz'], koelnclusters['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(koln_map)



# # let's start again with a clean copy of the map of San Francisco
# sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# # instantiate a mark cluster object for the incidents in the dataframe
# #plzs = plugins.MarkerCluster().add_to(koln_map)

# # loop through the dataframe and add each data point to the mark cluster
# for lat, lng, label, in zip(query_coordinates["longitude"], query_coordinates["latitude"], query_coordinates["n"]):
#     folium.CircleMarker(
#         [lat, lng],
#         radius=2,
#         popup=label,
#         color='blue',
#         fill=True,
#         fill_color='#3186cc',
#         fill_opacity=0.7,
#         parse_html=False).add_to(koln_map) 
    

# display map
koln_map.save("venues2.html")
koln_map
# }}}

koelnclusters.head()

# {{{
# 50733
# vs cluster 2
# }}}

forModel.head()

cluster2=forModel[forModel["Cluster Labels"]==2].groupby(["Cluster Labels"],as_index=False).mean()
cluster2

Nippes=forModel[forModel["plz"]=="50733"].groupby(["Cluster Labels"],as_index=False).mean()
diff=cluster2.transpose()-Nippes.transpose()
diff=diff.sort_values(by=[0],ascending=False)
diff.columns=["score"]
diff=diff[1:]
print("top 20 needs\n", diff[:20])
print("\ntop 20 trends in Nippes\n", diff[-20:])
