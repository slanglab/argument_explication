## Harnessing Toulmin’s theory for zero-shot argument explication

This is replication code to support the paper <a href="https://ankitaiisc.github.io/images/ArgEx_ACL_2024.pdf">Harnessing Toulmin’s theory for zero-shot argument explication</a> which is forthcoming in ACL 2024.

If you use this code or data, please cite our paper:
```
@article{gupta2024harnessing,
  author    = {Gupta, Ankita and Zuckerman, Ethan, and O'Connor, Brendan},
  title     = {Harnessing Toulmin’s theory for zero-shot argument explication},
  journal   = {ACL},
  year      = {2024},
  note      = {Forthcoming}
}
```

## Set-up
Follow these instructions to set-up your repository. You will need to download <a href="https://www.anaconda.com/">Anaconda</a> to run ```conda``` and ```python``` commands.

```
git clone https://github.com/slanglab/argument_explication.git
cd argument_explication
conda create -y --name argex python==3.9
conda activate argex
pip install -r requirements.txt
```


## Prior Argumentation Datasets for Evaluation
The evaluation data in the ```data``` folder come from the following sources:

1. <a href="https://github.com/maria-becker/IKAT-EN/tree/master/corpus">Implicit Knowledge in Argumentative Texts: An annotated Corpus</a>

2. <a href="https://github.com/peldszus/arg-microtexts">An annotated corpus of argumentative microtexts.</a>

3. <a href="https://github.com/UKPLab/argument-reasoning-comprehension-task">The Argument Reasoning Comprehension Task: Identification and Reconstruction of Implicit Warrants</a>

