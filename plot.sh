#!/bin/bash
cargo run --release > data.txt
python3 plot.py
