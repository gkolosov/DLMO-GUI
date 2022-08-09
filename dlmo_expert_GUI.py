
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import pickle
from IPython.display import display, clear_output
from ipywidgets import interact, interactive, fixed, interact_manual
import datetime as dt
# %matplotlib inline

#
# di = dict.fromkeys(np.unique(df['experiment_id']),0)
# for i,k in enumerate(di.keys()):
#     di[k]=i

# df = pd.read_csv(PATH)
# df['study_id']="X"
# df['melatonin']=df['melatonin'] + np.random.normal(loc=4, scale=0.5, size=len(df))
# df = df.replace({"experiment_id": di})
# df.to_csv(PATH, index=False)

#PATH_TRUE = "data/melatonin_data_N=261.csv"
PATH_TEST = "data/melatonin_data_N=261_test.csv"
PATH_SAVE = "data/expert_labels.pickle"

PATH = PATH_TEST

class State():
    def __init__(self, dlmo, dlmo_range_start, dlmo_range_end, confidence):
        self.dlmo = dlmo
        self.dlmo_range_start = dlmo_range_start
        self.dlmo_range_end = dlmo_range_end
        self.confidence = confidence


def get_data(experiment_id=None, path=PATH, n=0):
    df = pd.read_csv(path)
    experiment_id = experiment_id or np.unique(df['experiment_id'])[n]
    df = df.loc[df['experiment_id'] == experiment_id, ["clock_time", "melatonin"]]
    df['clock_time'] = pd.to_datetime(df['clock_time'])
    df = df.set_index("clock_time", drop=True)
    return df

def get_data_len(path=PATH):
    df = pd.read_csv(path)
    return len(np.unique(df['experiment_id']))

def make_box_layout():
    return widgets.Layout(
        border='solid 1px black',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px'
    )

class DlmoGui(widgets.HBox):

    def __init__(self, n=0):
        super().__init__()

        self.saved_states = dict()
        self.output = widgets.Output()
        with self.output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(8, 5))
            self.ax.autoscale(True)
        self.n = n
        self.max_len = get_data_len()
        self.define_immutable_widgets()
        self.update_profile(init=True)

    def get_data(self):
        self.df = get_data(n=self.n)
        self.start_date = self.df.index[0]
        self.end_date = self.df.index[-1]
        self.dates = pd.date_range(self.start_date, self.end_date, freq='5min')
        self.options = [(date.strftime('%H : %M'), date) for date in self.dates]

    def init_plot(self):
        date_form = mdates.DateFormatter('%H : %M')
        self.vline = self.ax.axvline(x=self.saved_states[self.n].dlmo, color='firebrick')
        self.vline_conf_min = self.ax.axvline(x=self.saved_states[self.n].dlmo_range_start, color='pink', linestyle='--')
        self.vline_conf_max = self.ax.axvline(x=self.saved_states[self.n].dlmo_range_end, color='pink', linestyle='--')
        self.line = self.ax.plot(self.df.index, self.df.values, linestyle='--', marker='o', color='royalblue')
        self.ax.xaxis.set_major_formatter(date_form)
        self.ax.relim()
        self.ax.autoscale()
        self.fig.suptitle('Profile {}/{}'.format(self.n+1, self.max_len), fontweight ="bold")
        self.output.layout = make_box_layout()
        # self.fig.canvas.toolbar_position = 'bottom'
        # self.fig.canvas.toolbar_visible = False


    def update_submit(self, b):
        with open(PATH_SAVE, 'wb') as handle:
            pickle.dump(self.saved_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.line[0].remove()
        self.vline_conf_min.remove()
        self.vline_conf_max.remove()
        self.vline.remove()
        self.prev_widget.disabled = True
        self.next_widget.disabled = True
        self.button_widget.disabled = True
        self.ax.axis('off')
        self.ax.text(x=self.dates[len(self.dates)//2], y=self.df.max()/2, s="Thank you !",
                  bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'none', 'pad': 1},
                  ha='center', va='center',fontsize=40)
        self.fig.canvas.toolbar_visible = False
        self.ax.relim()
        self.ax.autoscale()

    def update_left(self, b):
        self.n = self.n - 1
        self.update_profile()

    def update_right(self, b):
        self.n = self.n + 1
        self.update_profile()

    def update_dlmo(self, change):
        self.saved_states[self.n].dlmo = change.new
        self.vline.remove()
        self.vline = self.ax.axvline(x=change.new, color='firebrick')

    def update_confidence(self, change):
        self.saved_states[self.n].confidence = change.new

    def update_profile(self, init=False):

        if not init:
            self.line[0].remove()
            self.vline_conf_min.remove()
            self.vline_conf_max.remove()
            self.vline.remove()
            with open(PATH_SAVE, 'wb') as handle:
                pickle.dump(self.saved_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.get_data()
        if self.n not in self.saved_states:
            ### INIT soit start/end soit au milieu
            self.saved_states[self.n] = State(dlmo=self.end_date, dlmo_range_start=self.start_date,
                                              dlmo_range_end=self.end_date, confidence="N/A")
            #self.saved_states[self.n] = State(dlmo=self.dates[len(self.dates)//2], dlmo_range_start=self.dates[len(self.dates)//4],
             #                                 dlmo_range_end=self.dates[3*len(self.dates)//4], confidence="N/A")
        self.init_plot()
        self.define_widgets(options=self.options)
        self.observe_widgets()
        self.dlmo_widget_box = widgets.VBox([self.dlmo_range_widget, self.dlmo_widget])
        # self.helper_widget_box = widgets.HBox([self.confidence_widget, self.profile, self.button_widget])
        self.helper_widget_box = widgets.HBox(
            [self.confidence_widget, self.prev_widget, self.next_widget, self.button_widget])
        self.controls = widgets.VBox([self.dlmo_widget_box, self.helper_widget_box])
        self.controls.layout = make_box_layout()
        self.children = [self.controls, self.output]

        self.prev_widget.disabled = self.n == 0
        self.next_widget.disabled = self.n == self.max_len-1
        self.button_widget.disabled = self.n != self.max_len-1


    def update_dlmo_range(self, change):
        self.saved_states[self.n].dlmo_range_start = change.new[0]
        self.saved_states[self.n].dlmo_range_end = change.new[1]
        # self.ax.lines.pop(0)
        self.vline_conf_min.remove()
        self.vline_conf_min = self.ax.axvline(x=change.new[0], color='pink', linestyle='--')
        self.vline_conf_max.remove()
        self.vline_conf_max = self.ax.axvline(x=change.new[1], color='pink', linestyle='--')

    def observe_widgets(self):
        self.dlmo_range_widget.observe(self.update_dlmo_range, 'value')
        self.dlmo_widget.observe(self.update_dlmo, 'value')
        self.button_widget.on_click(self.update_submit)
        self.prev_widget.on_click(self.update_left)
        self.next_widget.on_click(self.update_right)
        self.confidence_widget.observe(self.update_confidence, 'value')

    def define_immutable_widgets(self):

        self.profile = widgets.Dropdown(
            options=[0, 1, 2],
            value=0,
            description='Profile:',
            disabled=False,
        )

        self.button_widget = widgets.Button(
            description='Submit All',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='check'
        )

        self.prev_widget = widgets.Button(
            description='Prev',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='fa-arrow-left'

        )

        self.next_widget = widgets.Button(
            description='Next',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='fa-arrow-right'

        )

    def define_widgets(self, options):
        self.confidence_widget = widgets.Dropdown(
            options=["N/A", 'High', 'Medium', 'Low'],
            value=self.saved_states[self.n].confidence,
            description='Confidence:',
            disabled=False,
        )
        self.dlmo_range_widget = widgets.SelectionRangeSlider(
            options=options,
            #index=(0, len(options) - 1),
            index=(self.dates.get_loc(self.saved_states[self.n].dlmo_range_start), self.dates.get_loc(self.saved_states[self.n].dlmo_range_end)),
            description='DLMO Range',
            orientation='horizontal',
            layout={'width': '900px'}
        )
        self.dlmo_widget = widgets.SelectionSlider(
            options=options,
            #index=len(options) - 1,
            index=self.dates.get_loc(self.saved_states[self.n].dlmo),
            description='DLMO Value',
            orientation='horizontal',
            layout={'width': '900px'},
        )

if __name__ == '__main__':
    df = pd.read_csv(PATH)
    df=df.loc[df['experiment_id'].isin(["3453HY52_1","3752A_1", "990153", "4190A_saliva", "68" ])]
    df.to_csv(PATH_TEST, index=False)
