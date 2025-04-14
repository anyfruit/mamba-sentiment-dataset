Mamba: Linear-Time Sequence Modeling for Sentiment Analysis

This project reimplements the Mamba model with TensorFlow and evaluates its performance on sentiment classification using the Sentiment140 dataset.

Overview

We're testing Mamba's capabilities on a dataset of 1.6 million tweets labeled for sentiment, a task requiring contextual understanding and nuance detection. Our goal is comparing Mamba directly against Transformer models to assess its efficiency-effectiveness claims.

Mamba offers several key advantages:

Linear-time processing versus the quadratic scaling of Transformer attention
Hardware-aware algorithm using recurrent scanning without materializing expanded states
Input-dependent selection mechanism that improves performance on discrete, information-dense data like text

Research Goals

Evaluate Mamba's sentiment analysis performance
Compare accuracy and speed against Transformer models
Analyze practical trade-offs for efficient NLP deployment

This project aims to provide insights into whether Mamba can match Transformer performance while delivering the computational efficiency promised in the original paper.
