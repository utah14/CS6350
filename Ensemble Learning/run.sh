#!/bin/bash
echo "running Adaboost..."
python3 Adaboost_bank.py
python3 Adaboost_credit.py

echo "running Boost and Bag..."
python3 boostAndBag.py

echo "finding bias and variance of trees/bagged trees..."
python3 bias.py
