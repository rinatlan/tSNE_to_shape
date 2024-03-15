# tSNE to shape
As explained in the [“Visualizing Data using t-SNE”](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf) article, the aim of the tSNE algorithm is to visualize data. 
In particular - convert data of high dimension to low dimension (2 or 3 dimensions). 
In this project I will focus on 2-dimension conversion. 

The tSNE algorithm takes into consideration only the relation between each point to the others. 
My goal is to emphasize this kind of behavior in the following way: reduce high dimensional data with tSNE to 
2-dimensional space, but with a certain shape. 

# How to use the code
There is a tutorial code which can be used to see the behavior I mentioned above.
In the `tutorial.ipynb` file you can find a guidance on how to use the code, play around with it, and see the results.


# Acknowledgments 
`my_t_sne.py` is based on the work of Alexander Fabisch, Christopher Moody and Nick Travers, and is used under the BSD 3-clause license.    
`shapes.py` and `meshes.py` are based on the work of [Jonathan Viquerat](https://github.com/jviquerat/shapes), and is used under the MIT license.    
tasic et al data and relatd files are based on files from [https://portal.brain-map.org/atlases-and-data/rnaseq](https://portal.brain-map.org/atlases-and-data/rnaseq).    
Special thanks to Ela Fallik and the Friedman Lab for their guidance and support throughout this project. 
