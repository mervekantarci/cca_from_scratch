# Connected Component Labeling from Scratch

This repository counts the components in an image by applying basic image processing techniques and running connected component labeling algorithm.
The algortihm is implemented from scratch.

**Usage**
```
-h, --help    show this help message and exit
-file FILE    path to input file
--save        processed image will be saved
--no_display  processed image will not be displayed
```

**Example Command**
```
python main.py -file sample1.jpg --save --no_display
```

Input file             |  Processed file
:-------------------------:|:-------------------------:
![](https://github.com/mervekantarci/cca_from_scratch/blob/main/sample1.jpg)  |  ![](https://github.com/mervekantarci/cca_from_scratch/blob/main/sample1_processed.jpg)

**Output**
```
Number of components: 9
```
