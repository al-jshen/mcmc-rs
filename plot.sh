#!/bin/bash
RUSTFLAGS="-C target-cpu=native" cargo run --release > data.py
python3 plot.py
