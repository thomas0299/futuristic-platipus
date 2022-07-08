#importing all relevant packages for whole capstone project

#for data wrangling
import numpy as np
import pandas as pd
from datetime import datetime, date

#for visualisations
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Image

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#for api requests
import requests
import os
from sodapy import Socrata

#for modelling

#For the sake of output, we disable warnings. All warnings related to the version of libraries
import warnings
warnings.filterwarnings('ignore')