# Explainable ML Experiments

Commands:
- Visualize activations: `python visualize_activations.py --checkpoint mnist_cnn.pt --sample-index 0 --output-dir activations --show`
- Visualize ALE plots: `python ale_mnist_cnn.py --checkpoint mnist_cnn.pt --num-features 16 --class-idx 0 --output-dir ale_out`