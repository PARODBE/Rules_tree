[![Python](https://img.shields.io/pypi/pyversions/torchquad)](https://img.shields.io/pypi/pyversions/torchquad)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://img.shields.io/badge/license-GPLv3-blue)

This library, using very simple terminology, aims to make the rules for categorical variables obtained from a classification tree through Scikit-Learn easier to understand.

For example, in the Titanic dataset, if we extract the textual rules provided by Scikit-Learn, we see that:

<p align="center">
  <img src="https://github.com/PARODBE/Rules_tree/blob/main/Scikit_rules.png" alt="Cover Page">
</p>

For the variable Sex, it specifies <= 0.5 instead of Male or Female. In this case, the variable is binary. If we now look at a continuous variable, such as Class, we can see that a rule is established for <= 1.5. In both cases, the rules provided are not very interpretable. However, if we use this library, we can see the following rules:

<p align="center">
  <img src="https://github.com/PARODBE/Rules_tree/blob/main/Rules_rules.png" alt="Cover Page" width="500" height="200">
</p>

As we can see, it starts by splitting the rules according to the variable Sex, which matches what was obtained with Scikit-Learn, and likewise, we can see that...
