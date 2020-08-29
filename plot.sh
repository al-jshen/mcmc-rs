#!/bin/bash
cargo run --release > data.py
python3 plot.py
