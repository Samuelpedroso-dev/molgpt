# MolGPT

A small GPT that writes molecules as SMILES strings, with an interface to explore what it invents as a network of similar molecules, in 2D and in 3D. Early version 0.1, more coming.

Live demo and landing page are served from this repository on Vercel. The interface is a single offline HTML file in `interface/index.html`.

## What is in this repository

`molgpt.py` holds the full code for data, model, training, generation and evaluation. `molgpt.ipynb` is the same code as a notebook for Colab with a GPU. `model/molgpt.pt` is the checkpoint of the small model. `results/` keeps the metrics and the training log. `interface/index.html` is the explorer, and `index.html` at the root is the landing page.

## The small model

The model has 0.81 million parameters, 4 layers, 4 heads and an embedding size of 128. It was trained for 2600 steps on ZINC250k, in about 28 minutes on a 2 core CPU. The final loss was about 0.77 on both training and validation, so it learned without memorizing.

The model reads a molecule as a sequence of tokens that starts with a begin marker and two control tokens, one for a logP bin and one for a QED bin. At generation time you give the control tokens and the model writes the rest. RDKit then checks every result.

## Results

4000 molecules were generated and compared with a real ZINC250k sample, measured with RDKit.

| Measure | GPT | Real ZINC |
|---|---|---|
| Valid molecules | 56% | 100% |
| Unique among valid | 100% | |
| Diversity | 0.877 | 0.881 |
| Mean QED | 0.730 | 0.738 |
| Mean logP | 2.43 | 2.38 |
| Pass Lipinski rule | 96.5% | 98.7% |
| PAINS alerts | 2.1% | 3.1% |

Validity is the weak spot, which is expected for a model this small with such short training. The control by logP bin works. Asking from the lowest bin to the highest, the mean logP of the returned molecules rose each time, from 0.75 to 4.17. Full numbers are in `results/metrics.json`.

## Run it

```
pip install rdkit torch pandas numpy matplotlib
python molgpt.py cpu     # small model, about 28 minutes on 2 CPU cores
python molgpt.py         # full model, meant for a GPU
```

The script downloads ZINC250k by itself. The full preset has about 5 million parameters and is meant for a GPU, for example a free Colab T4. The time estimate for GPU was not measured.

## Note on the files

The numbers in `results/` and the recorded writing in the interface come from the first training run. The checkpoint in `model/` comes from a second run with the same preset and seed, made after the first checkpoint was overwritten by mistake. The two runs are very close but not identical, so small differences are expected if you evaluate the checkpoint yourself.

## Limits

The model only knows the shape of drug like molecules from ZINC250k. It knows nothing about protein targets, binding affinity or toxicity, so what it writes are starting candidates and not drugs. The interface shows recorded molecules and recorded writing from the small model and does not run the model in the browser. The next step is an affinity model, trained on Davis or KIBA, acting as a judge for the generated molecules, followed by reinforcement fine tuning and docking of the best candidates into a real target.

## License

MIT, see `LICENSE`.
