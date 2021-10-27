# 🧠 kendrite

A framework for attentive explainable deep learning on tabular data

---

# Problem

What problem do we solve?

# Solution

How do we solve it?

# Showcase

How does it work?

# TabNet overview

> Attention transformer blocks

* Instance-wise feature selection
* Built-in explainability derived from masks
* Effcient learning capacity by only focusing on most important features
* Masks can be avereaged and give instance-wise interpretatio for the models output.
* A feature that has been masked a lot (not used) has low importance for the model and vica-versa.

> Feature transformer blocks

* MLP block with glated linearu unit activation
* Shared layers across different steps

> Sequential steps

* Repeat blocks, mimics ensembling/boosting
* All steps contribute equally to the final mapping layer (regression or classification) for the output

> Parameters

Main hyperparameters are `n_d`, `n_a` and `n_steps`.
