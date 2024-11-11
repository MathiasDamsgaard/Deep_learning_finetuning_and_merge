import argparse
import os
from baseline_model import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("BM", help="Baseline model")
    parser.add_argument("epochs", help="Number of epochs")

    args = parser.parse_args()

    if args.BM == "baseline_model":
        train_model(int(args.epochs))
    

if __name__ == "__main__":
    main()
    
