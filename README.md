# NeuroDecipher

This repo hosts the codebase for reproduce the results for the ACL paper [Neural Decipherment via Minimum-Cost Flow: from Ugaritic to Linear B](http://people.csail.mit.edu/j_luo/assets/publications/NeuroDecipher.pdf). Still cleaning up some part of the code. Stay tuned for updates.

# Data
Data for linear B and Ugaritic are included in the `data` folder:
* `uga-heb.no_spe.cog` is the entire Ugaritic-Hebrew data obtained from [Ben Snyder](http://people.csail.mit.edu/bsnyder/papers/bsnyder_acl2010.pdf). `no_spe` stands for no special symbols since the original file contains special symbols that mark the morphological segmentations and affixes.
* `uga-heb.small.no_spe.cog` is the exact random subset of Ugaritic data I used in the paper for training the model. Around one tenth of the original file.
* `linear_b-greek.cog` is the linear B data used in the paper. `notebooks/Linear_b_simplified.ipynb` is the same notebook I used for preparing the linear B data.
* `linear_b-greek.names.cog` is the linear B data that only included names on the Greek side.

Note that you might need to install [fonts](https://www.google.com/get/noto/#sans-linb) in order to render Linear B scripts properly in your computer.

