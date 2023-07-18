import sys
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import random
from scipy.optimize import fsolve
from scipy.optimize import minimize
import math
from scipy.stats import norm
from statistics import NormalDist
from scipy.integrate import quad
from itertools import product


print("task %d is done" %int(sys.argv[1]))

txt_file = open("test.txt", "a")
txt_file.write("%d" %int(sys.argv[1]))
txt_file.close()
