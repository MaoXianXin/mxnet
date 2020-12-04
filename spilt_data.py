import splitfolders  # or import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("/home/mao/Downloads/datasets/minc-2500/images", output="/home/mao/Downloads/datasets/minc-2500", seed=1337, ratio=(0.8, 0.2), group_prefix=None) # default values