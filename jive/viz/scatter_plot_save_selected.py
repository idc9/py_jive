from bokeh.models import ColumnDataSource, CustomJS, CategoricalColorMapper
from bokeh.io import push_notebook, show
from bokeh.plotting import figure
from bokeh.palettes import d3

import numpy as np


"""
from scatter_plot_save_selected import setup_data,get_handle,get_save_selected
from bokeh.io import output_notebook

import numpy as np
x = np.random.normal(size=100)
y = np.random.normal(size=100)
names = list(range(100))

output_notebook()
source, model = setup_data(x=x, y=x, names=names)
figkwds = dict(plot_width=800, plot_height=800,
               x_axis_label='x',
               y_axis_label='y',
               tools="pan,lasso_select,box_select,reset,help")
handle = get_handle(source, figkwds)
saved_selected = get_save_selected(handle, model)

model.to_df()
"""


def set_save_selected_callback(source):
    source.callback = CustomJS(code="""
        // Define a callback to capture errors on the Python side
        function callback(msg){
            console.log("Python callback returned unexpected message:", msg)
        }
        callbacks = {iopub: {output: callback}};

        // Select the data
        var inds = cb_obj.selected['1d'].indices;
        var d1 = cb_obj.data;
        // var x = []
        // var y = []
        var names = []
        for (i = 0; i < inds.length; i++) {
            // x.push(d1['x'][inds[i]])
            // y.push(d1['y'][inds[i]])
            names.push(d1['name'][inds[i]])
        }

        // Generate a command to execute in Python
        data = {
            // 'x': x,
            // 'y': y,
            'names': names,
        }
        var data_str = JSON.stringify(data)
        var cmd = "saved_selected(" + data_str + ")"

        // Execute the command on the Python kernel
        var kernel = IPython.notebook.kernel;
        kernel.execute(cmd, callbacks, {silent : false});
    """)

    return source


def setup_data(x, y, names=None, classes=None):
    if names is None:
        names = list(range(len(x)))

    data = {'name': names, 'x': x, 'y': y}
    if classes is not None:
        data['class'] = classes
    source = ColumnDataSource(data)
    source = set_save_selected_callback(source)

    model = ColumnDataSource(data=dict(
        # x=[],
        # y=[],
        names=[],
    ))
    return source, model


def get_handle(source, figkwds=None):

    if figkwds is None:
        figkwds = dict(plot_width=500, plot_height=300,
                       x_axis_label='x', y_axis_label='y',
                       tools="pan,lasso_select,box_select,reset,help")

    if 'class' in source.data.keys():
        class_labels = list(set(source.data['class']))

        palette = d3['Category10'][len(class_labels)]
        color_map = CategoricalColorMapper(factors=class_labels,
                                           palette=palette)

        color = {'field': 'class', 'transform': color_map}
        legend = 'class'
    else:
        color = 'blue'
        legend = None

    p1 = figure(active_drag="lasso_select", **figkwds)
    p1.scatter('x', 'y', source=source, alpha=0.8,
               color=color, legend=legend)

    handle = show(p1, notebook_handle=True)

    return handle


def get_save_selected(handle, model):

    selected = dict()

    def saved_selected(values):
        # x_sel = np.array(values['x'])
        # y_sel = np.array(values['y'])
        names_sel = np.array(values['names'])

        # data = {'x': x_sel,  'y': y_sel, 'names': names_sel}
        data = {'names': names_sel}
        model.data.update(data)
        # Update the selected dict for further manipulation
        selected.update(data)
        # Update the drawing
        push_notebook(handle=handle)

    return saved_selected
