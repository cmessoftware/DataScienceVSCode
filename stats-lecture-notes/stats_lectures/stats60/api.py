from PL_density import *
from conv import *

stats60_examples = {}
import dice, roulette, monty_hall, marbles
for E in [dice.examples, roulette.examples, monty_hall.examples]:
    for k in E:
        stats60_examples[k] = E[k]

from correlation import pearson_lee
from gender_bias import UCB, UCB_female, UCB_male
from sample import Sample
from testing import BloodPressure

stats60_figsize = (5.5,5.5)

import matplotlib
matplotlib.rcParams['figure.figsize'] = stats60_figsize

from tables import *

from probability import *
