# -*- coding: utf-8 -*-
from functools import lru_cache

from os import listdir
from os.path import dirname, join

import numpy as np
import pandas as pd
import pickle
import sqlite3

from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox, layout
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Div, Legend, Spacer, Range1d
from bokeh.models.widgets import Select, TextInput, CheckboxGroup, RangeSlider
from bokeh.plotting import figure

from io import StringIO
import base64

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

# creat dictionaries for plotting selections
with open(join(dirname(__file__), 'axis_map.pkl'), 'rb') as handle:
    axis_map = pickle.load(handle)
with open(join(dirname(__file__), 'library_map.pkl'), 'rb') as handle:
    library_map = pickle.load(handle)
with open(join(dirname(__file__), 'reverse_library_map.pkl'), 'rb') as handle:
    reverse_library_map = pickle.load(handle)

for pc in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']:
    axis_map[pc] = pc

# open database connections
conn = sqlite3.connect(join(dirname(__file__), 'commercial_fragment_libraries.sqlite3'))
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(data)")
fields = cursor.fetchall()
column_names = [x[1] for x in fields]

pca_map = {"Autocorr2D" : "CalcAUTOCORR2D-pca",
            "Autocorr3D" : "CalcAUTOCORR3D-pca",
            "GETAWAY" : "CalcGETAWAY-pca",
            "MORSE" : "CalcMORSE-pca",
            "RDF" : "CalcRDF-pca",
            "WHIM" : "CalcWHIM-pca",
            "Atom Pair Fingerprint" : "GetHashedAtomPairFingerprintAsBitVect-pca",
            "Topological Torsion Fingerprint" : "GetHashedTopologicalTorsionFingerprintAsBitVect-pca",
            "MACCS Keys Fingerprint" : "GetMACCSKeysFingerprint-pca",
            "Layered Fingerprint" : "LayeredFingerprint-pca",
            "Pattern Fingerprint" : "PatternFingerprint-pca",
            "RDKit Fingerprint" : "RDKFingerprint-pca",
            "Mol2Vec Features" : "mol2vec-pca"}

# Create selection controls
colors = ['grey', 'yellow', 'orange', 'red', 'green', 'cyan', 'blue', 'purple']
color_choice = Select(title="Next Circle Plot Color", options=colors, value='grey')
x_axis = Select(title="X Axis", options=sorted(list(axis_map.keys())), value="PC1")
y_axis = Select(title="Y Axis", options=sorted(list(axis_map.keys())), value="PC2")
size_by = Select(title="Circle Size By Attribute", options=sorted(list(axis_map.keys())), value='MolWt')
alpha_by = Select(title="Circle Alpha By Attribute", options=sorted(list(axis_map.keys())), value='MolLogP')
pca_select = Select(title="Principle Component Analysis", options=sorted(list(pca_map.keys())), value='RDKit Fingerprint')

# instantiate plotting dataframe
compounds = pd.DataFrame(columns=['ID', 'color', x_axis.value, y_axis.value, size_by.value, alpha_by.value, 'png'])

# Create library selection check boxes
library_names = list(library_map.keys())
ordered_library_names = list(np.sort(library_names))
library = CheckboxGroup(labels=ordered_library_names, active=[])

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], color=[], alpha=[], size=[], label=[], png=[]))

# Create scatter plot
TOOLTIPS=[
        ("label", "@label"),
        ("size_real", "@size_real"),
        ("alpha_real", "@alpha_real"),
        ("png", "@png")
]

hover = HoverTool(tooltips="""
        <div>
                <div>
                        <img
                                src="@png" height="150" alt="@png" width="150"
                                style="float: left; margin: 0px 15px 15px 0px;"
                                border="0"
                        ></img>
                </div>
                <div>alpha = @alpha_real</div>
                <div>size = @size_real</div>
                <div>@label</div>
        </div>
        """
)

p = figure(plot_height=600, plot_width=700, title="", tools=['reset, box_zoom, wheel_zoom, zoom_out, pan, save', hover])
r = p.scatter(x="x", y="y", source=source, color="color", line_color=None, size="size", fill_alpha="alpha", legend="label")

# create the horizontal histogram
hhist, hedges = np.histogram(source.data['x'], bins=20)
hzeros = np.zeros(len(hedges)-1)
hmax = max(hhist)*1.1

LINE_ARGS = dict(color="#3A5785", line_color=None)

ph = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200, x_range=p.x_range,
            min_border=10, min_border_left=50, y_axis_location="right")

ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"

hhgrey = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="grey", alpha=0.4)
hhyell = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="yellow", alpha=0.4)
hhoran = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="orange", alpha=0.4)
hhred = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="red", alpha=0.4)
hhgree = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="green", alpha=0.4)
hhcyan = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="cyan", alpha=0.4)
hhblue = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="blue", alpha=0.4)
hhpurp = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="purple", alpha=0.4)

# create the vertical histogram
vhist, vedges = np.histogram(source.data['y'], bins=20)
vzeros = np.zeros(len(vedges)-1)
vmax = max(vhist)*1.1

pv = figure(toolbar_location=None, plot_width=200, plot_height=p.plot_height,
            y_range=p.y_range, min_border=10, y_axis_location="right")

pv.ygrid.grid_line_color = None
pv.xaxis.major_label_orientation = np.pi/4
pv.background_fill_color = "#fafafa"

vhgrey = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="grey", alpha=0.4)
vhyell = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="yellow", alpha=0.4)
vhoran = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="orange", alpha=0.4)
vhred = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="red", alpha=0.4)
vhgree = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="green", alpha=0.4)
vhcyan = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="cyan", alpha=0.4)
vhblue = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="blue", alpha=0.4)
vhpurp = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="purple", alpha=0.4)

hhs = [hhgrey, hhyell, hhoran, hhred, hhgree, hhcyan, hhblue, hhpurp]
vhs = [vhgrey, vhyell, vhoran, vhred, vhgree, vhcyan, vhblue, vhpurp]

# bokeh plottomg methods
def select_libraries(attr, old, new):
    global compounds
    for i in range(len(library_names)):
        ix = library_names.index(ordered_library_names[i])
        substring = '-'.join(list(library_map.values())[ix].split('-')[:-1])+'-'
        if i not in library.active:
            if 'Prestw' in substring:
                compounds = compounds[compounds.ID.str.contains('Prestw') == False]
            else:   
                compounds = compounds[compounds.ID.str.contains(substring) == False]
        if i in library.active:
            strings = ['-'.join(j.split('-')[:-1])+"-" for j in compounds.ID]
            if substring not in strings:
                resultDf = pd.read_sql_query('SELECT ID,zinc FROM data WHERE ID LIKE "'+substring+'%"', conn)
                png = [str]*len(resultDf)
                for z, ix in zip(resultDf.zinc, range(len(resultDf.zinc))):
                    png[ix] = "http://zinc15.docking.org/substances/"+z+".png"
                resultDf['png'] = png
                resultDf = resultDf[['ID', 'png']]
                for att in [x_axis.value, y_axis.value, size_by.value, alpha_by.value]:
                    selected = axis_map[att]
                    if 'PC' in att:
                        tmpDf = pd.read_pickle(join(dirname(__file__), pca_map[pca_select.value]+".pkl"))[['ID', att]]
                        resultDf = pd.merge(resultDf, tmpDf, on=['ID'])
                    else:
                        tmpDf = pd.read_sql_query('SELECT ID,"'+selected+'" FROM data WHERE ID LIKE "'+substring+'%"', conn)
                        resultDf = pd.merge(resultDf, tmpDf, on=['ID'])
                resultDf['color'] = [color_choice.value]*len(resultDf)
                compounds = compounds.append(resultDf, sort=False)
    update()

def select_x_axis(attr, old, new):
    global compounds
    att = x_axis.value
    if att not in list(compounds):
        resultDf = pd.DataFrame(columns=['ID', att])
        for i in range(len(library_names)):
            ix = library_names.index(ordered_library_names[i])
            substring = '-'.join(list(library_map.values())[ix].split('-')[:-1])+'-'
            if i not in library.active:
                compounds = compounds[compounds.ID.str.contains(substring) == False]
            if i in library.active:
                selected = axis_map[att]
                strings = ['-'.join(j.split('-')[:-1])+"-" for j in compounds.ID]
                if 'PC' in att:
                    tmpDf = pd.read_pickle(join(dirname(__file__), pca_map[pca_select.value]+".pkl"))[['ID', att]]
                else:
                    tmpDf = pd.read_sql_query('SELECT ID,"'+selected+'" FROM data WHERE ID LIKE "'+substring+'%"', conn)
                resultDf = resultDf.append(tmpDf)
        actives = [axis_map[x_axis.value], axis_map[y_axis.value], axis_map[alpha_by.value], axis_map[size_by.value], 'ID', 'png', 'color']
        for l in list(compounds):
            if l not in actives:
                compounds = compounds.drop(l, 1)
        compounds = pd.merge(compounds, resultDf, on=['ID'])
    update()

def select_y_axis(attr, old, new):
    global compounds
    att = y_axis.value
    if att not in list(compounds):
        resultDf = pd.DataFrame(columns=['ID', att])
        for i in range(len(library_names)):
            ix = library_names.index(ordered_library_names[i])
            substring = '-'.join(list(library_map.values())[ix].split('-')[:-1])+'-'
            if i not in library.active:
                compounds = compounds[compounds.ID.str.contains(substring) == False]
            if i in library.active:
                strings = ['-'.join(j.split('-')[:-1])+"-" for j in compounds.ID]
                selected = axis_map[att]
                if 'PC' in att:
                    tmpDf = pd.read_pickle(join(dirname(__file__), pca_map[pca_select.value]+".pkl"))[['ID', att]]
                else:
                    tmpDf = pd.read_sql_query('SELECT ID,"'+selected+'" FROM data WHERE ID LIKE "'+substring+'%"', conn)
                resultDf = resultDf.append(tmpDf)
        actives = [axis_map[x_axis.value], axis_map[y_axis.value], axis_map[alpha_by.value], axis_map[size_by.value], 'ID', 'png', 'color']
        for l in list(compounds):
            if l not in actives:
                compounds = compounds.drop(l, 1)
        compounds = pd.merge(compounds, resultDf, on=['ID'])
    update()

def select_alpha(attr, old, new):
    global compounds
    att = alpha_by.value
    if att not in list(compounds):
        resultDf = pd.DataFrame(columns=['ID', att])
        for i in range(len(library_names)):
            ix = library_names.index(ordered_library_names[i])
            substring = '-'.join(list(library_map.values())[ix].split('-')[:-1])+'-'
            if i not in library.active:
                compounds = compounds[compounds.ID.str.contains(substring) == False]
            if i in library.active:
                strings = ['-'.join(j.split('-')[:-1])+"-" for j in compounds.ID]
                selected = axis_map[att]
                if 'PC' in att:
                    tmpDf = pd.read_pickle(join(dirname(__file__), pca_map[pca_select.value]+".pkl"))[['ID', att]]
                else:
                    tmpDf = pd.read_sql_query('SELECT ID,"'+selected+'" FROM data WHERE ID LIKE "'+substring+'%"', conn)
                resultDf = resultDf.append(tmpDf)
        actives = [axis_map[x_axis.value], axis_map[y_axis.value], axis_map[alpha_by.value], axis_map[size_by.value], 'ID', 'png', 'color']
        for l in list(compounds):
            if l not in actives:
                compounds = compounds.drop(l, 1)
        compounds = pd.merge(compounds, resultDf, on=['ID'])
    update()

def select_size(attr, old, new):
    global compounds
    att = size_by.value
    if att not in list(compounds):
        resultDf = pd.DataFrame(columns=['ID', att])
        for i in range(len(library_names)):
            ix = library_names.index(ordered_library_names[i])
            substring = '-'.join(list(library_map.values())[ix].split('-')[:-1])+'-'
            if i not in library.active:
                compounds = compounds[compounds.ID.str.contains(substring) == False]
            if i in library.active:
                strings = ['-'.join(j.split('-')[:-1])+"-" for j in compounds.ID]
                selected = axis_map[att]
                if 'PC' in att:
                    tmpDf = pd.read_pickle(join(dirname(__file__), pca_map[pca_select.value]+".pkl"))[['ID', att]]
                else:
                    tmpDf = pd.read_sql_query('SELECT ID,"'+selected+'" FROM data WHERE ID LIKE "'+substring+'%"', conn)
                resultDf = resultDf.append(tmpDf)
        actives = [axis_map[x_axis.value], axis_map[y_axis.value], axis_map[alpha_by.value], axis_map[size_by.value], 'ID', 'png', 'color']
        for l in list(compounds):
            if l not in actives:
                compounds = compounds.drop(l, 1)
        compounds = pd.merge(compounds, resultDf, on=['ID'])
    update()

def select_pca_table(attr, old, new):
    global compounds
    actives = [x_axis.value, y_axis.value, alpha_by.value, size_by.value]
    for att, ix in zip(actives, range(len(actives))):
        if 'PC' in att:
            if ix == 0:
                compounds = compounds.drop(x_axis.value, 1)
                select_x_axis(attr, old, new)
            if ix == 1:
                compounds = compounds.drop(y_axis.value, 1)
                select_y_axis(attr, old, new)
            if ix == 2:
                compounds = compounds.drop(alpha_by.value, 1)
                select_alpha(attr, old, new)
            if ix == 3:
                compounds = compounds.drop(size_by.value, 1)
                select_size(attr, old, new)

def update():
    global compounds
    compounds = compounds.drop_duplicates()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]
    alpha = axis_map[alpha_by.value]
    size = axis_map[size_by.value]
    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "%d compounds plotted" % len(compounds)
    if len(compounds) > 0:
        compounds = compounds.dropna(axis=1, how = 'all')
        compounds = compounds.dropna()
        for c in list(compounds):
            compounds[c] = pd.to_numeric(compounds[c], errors="ignore")
        lib_names = []
        for id in compounds.ID:
            if 'Prestw-FRAG-' in id:
                lib_names.append('Prestw-Frag-')
            else:
                lib_names.append(reverse_library_map['-'.join(id.split('-')[:-1])+'-'])        
        source.data = dict(
            x=compounds[x_name],
            y=compounds[y_name],
            size=np.array(10*min_max_scaler.fit_transform(np.array(compounds[size]).reshape(-1, 1))).T[0]+5,
            alpha=np.array(min_max_scaler.fit_transform(np.array(compounds[alpha]).reshape(-1, 1))+0.1).T[0]/1.1,
            color=compounds["color"],
            label=lib_names,
            png=compounds["png"],
            alpha_real=compounds[alpha],
            size_real=compounds[size]
        )

    ph.x_range=p.x_range
    pv.y_range=p.y_range
    tmp_df = pd.DataFrame(source.data)
    
    if len(tmp_df) > 0:
        for c, v, h in zip(colors, vhs, hhs):
            x = tmp_df[tmp_df.color == c].x
            y = tmp_df[tmp_df.color == c].y
            
            hhist, hhedges = np.histogram(x, bins=20)
            h.data_source.data["top"]   =  hhist
            h.data_source.data["right"]   =  hhedges[1:]
            h.data_source.data["left"]   =  hhedges[:-1]

            vhist, vedges = np.histogram(y, bins=20)
            v.data_source.data["right"] =  vhist
            v.data_source.data["top"] =  vedges[1:]
            v.data_source.data["bottom"] =  vedges[:-1]

# define callbacks
x_axis.on_change('value', select_x_axis)
y_axis.on_change('value', select_y_axis)
size_by.on_change('value', select_size)
alpha_by.on_change('value', select_alpha)
pca_select.on_change('value', select_pca_table)

library.on_change('active', select_libraries)

# page layout
desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=1200)
footer = Div(text=open(join(dirname(__file__), "footer.html"), encoding="utf8").read(), width=1200)
sizing_mode = 'fixed'

main_controls = column(x_axis, y_axis, alpha_by, size_by, color_choice, pca_select, width=400)
fig = column(row(p, pv), row(ph, Spacer(width=200, height=200)))
library_box = column(widgetbox(library))
main_row = row(library_box, column(row(main_controls, fig), footer))

l = layout([
    [desc],
    [main_row]
], sizing_mode=sizing_mode)

update()

curdoc().add_root(l)
curdoc().title = 'cfbsA'
