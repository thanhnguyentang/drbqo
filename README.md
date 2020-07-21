
## Introduction 

This is the code for our AISTATS 2020 paper ["Distributionally Robust Bayesian Quadrature Optimization"](https://arxiv.org/abs/2001.06814). 


## Dependencies 

* Numpy  
* Scipy  
* [GPy](https://sheffieldml.github.io/GPy/)  
* [GPyOpt](https://github.com/SheffieldML/GPyOpt)
* [DIRECT](https://pypi.org/project/DIRECT/)    


## Run 

Within the main directory `drbqo`: 

```
python -m examples.run_drbqo_synthetic 

```


## Reference  

```

@InProceedings{pmlr-v108-nguyen20a,
  title = 	 {Distributionally Robust Bayesian Quadrature Optimization},
  author = 	 {Nguyen, Thanh and Gupta, Sunil and Ha, Huong and Rana, Santu and Venkatesh, Svetha},
  booktitle = 	 {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics},
  pages = 	 {1921--1931},
  year = 	 {2020},
  editor = 	 {Chiappa, Silvia and Calandra, Roberto},
  volume = 	 {108},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Online},
  month = 	 {26--28 Aug},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v108/nguyen20a/nguyen20a.pdf},
  url = 	 {http://proceedings.mlr.press/v108/nguyen20a.html},
  abstract = 	 {Bayesian quadrature optimization (BQO) maximizes the expectation of an expensive black-box integrand taken over a known probability distribution. In this work, we study BQO under distributional uncertainty in which the underlying probability distribution is unknown except for a limited set of its i.i.d samples. A standard BQO approach maximizes the Monte Carlo estimate of the true expected objective given the fixed sample set. Though Monte Carlo estimate is unbiased, it has high variance given a small set of samples; thus can result in a spurious objective function. We adopt the distributionally robust optimization perspective to this problem by maximizing the expected objective under the most adversarial distribution. In particular, we propose a novel posterior sampling based algorithm, namely distributionally robust BQO (DRBQO) for this purpose. We demonstrate the empirical effectiveness of our proposed framework in synthetic and real-world problems, and characterize its theoretical convergence via Bayesian regret.}
}

```

## Contact   

[Thanh Tang Nguyen](https://thanhnguyentang.github.io/). 

