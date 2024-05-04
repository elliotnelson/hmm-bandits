import argparse
#import jsonlines
from typing import List

from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np
import math
import os
import re
import trlx
import torch
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
