# A transfer learning metamodel using artificial neural networks applied to natural convection flows in enclosures

While buoyancy is what drives natural convection, the performance of a natural convection system can be affected by other physical phenomena and different factors such as geometry, boundary conditions, and the behavior of a fluid. Data-driven metamodels that govern real-world natural convection systems need datasets that cover the entire feature space. Moreover, some unforeseen features may become active under a different process or after a system is redesigned. Among other limiting factors, generating a new full set of simulations or experiments is time-consuming. Our methodology using TL with DNN can flexibly adapt to the expansion of the feature space when a natural convection system becomes more complicated and needs to be described more precisely.
We employed a transfer learning technique to predict the Nusselt number for natural convection flows in enclosures. Specifically, we considered the benchmark problem of a two-dimensional square enclosure with isolated horizontal walls and vertical walls at constant temperatures. The Rayleigh and Prandtl numbers are sufficient parameters to simulate this problem numerically. We adopted two approaches to this problem: Firstly, we made use of a multi-grid dataset in order to train our artificial neural network in a cost-effective manner. By monitoring the training losses for this dataset, we detected any significant anomalies that stemmed from an insufficient grid size, which we further corrected by altering the grid size or adding more data. Secondly, we sought to endow our metamodel with the ability to account for additional input features by performing transfer learning using deep neural networks. We trained a neural network with a single input feature (Rayleigh) and extended it to incorporate the effects of a second feature (Prandtl). We also considered the case of hollow enclosures, demonstrating that our learning framework can be applied to systems with higher physical complexity, while bringing the computational and training costs down.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

```
Python modules os, sys, time, winsound, numpy, pandas, matplotlib.pyplot
tensorflow, tensorflow_docs.modeling, tensorflow_docs.plots
tensorflow.keras (layers, load_model, Model, utils, backend)
sklearn.model_selection
```

## Authors

* **Majid Ashouri**

See also the list of [contributors](https://github.com/engdatasci/NCTL/contributors) who participated in this project.
