# Data format
Each `.cog` file is essentially a tsv file, where each column corresponds to the words in one language. Words in the same row are considered cognates. If for one word, there is no corresponding cognate in another language, `_` is used to fill the cell. If multiple cognates are available for the same word, '|' is used to separate them.
