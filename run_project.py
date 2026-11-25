"""Convenience command runner: python run_project.py [validate|eda|train]."""
import argparse
from src.config import DATA_PATH
from src.data_validation import load_data
from src.eda import run_eda
from src.train import train

parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["validate","eda","train"],default="validate",nargs="?")
args=parser.parse_args()
if args.command=="validate": print(load_data().shape)
elif args.command=="eda": print(run_eda())
else: print(train())
