**ReactionDataExtractor** is a toolkit for automatic extraction of data from chemical reaction schemes.

This guide provides a quick tour through ReactionDataExtractor concepts and functionality.

## Features

- Automatic extraction of chemical reaction schemes
- Segmentation of reaction arrows, conditions, diagrams and labels
- Optical recognition of chemical structures and text, parsing of reaction condiitions
- Whole-reaction-scheme recovery and conversion into a machine-readable format
- High-throughput capabilities
- Direct extraction from image files
- PNG, GIF, JPEG, TIFF image format support


# Installation

There are three ways to install the package. The first option is recommended due to its simplicity.

### Option 1 - Running ReactionDataExtractor inside a Docker container - Recommended
This is the simplest and most universal approach making the tool much more portable. Details on the installation process can be found [here](https://hub.docker.com/r/dmwil/reactiondataextractor).

### Option 2 - Installation via Conda

Anaconda Python is a self-contained Python environment that is useful for scientific applications.

First, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), which contains a complete Python distribution alongside the conda package manager.

Next, go to the command line terminal and create a **working environment** by typing

    conda create --name <my_env> python=3.6
    
Once this is created, enter this environment with the command

    conda activate <my_env>

and install ReactionDataExtractor by typing

    conda install -c dmwil ReactionDataExtractor
    
This command installs ReactionDataExtractor and all its dependencies from the author's channel.
This includes [**pyosra**](https://github.com/dmw51/pyosra), the Python wrapper for the OSRA toolkit, and [**ChemDataExtractor**](http://chemdataextractor.org), a library used for natural language processing.

Finally, download the **data files** for [ChemDataExtractor](http://chemdataextractor.org). These files contain the machine learning models, dictionaries and word clusters that ChemDataExtractor uses. This is done with the following command:

    cde data download
    
*This method of installation is currently supported on ubuntu machines only*

### Option 3 - Installation from source

We **strongly recommend** installation via conda whenever possible as all the dependencies are automatically handled. 
If this cannot be done, advanced users are invited to compile the code from source. These code repositories can be found at the locations below:

1. [Pyosra](https://github.com/edbeard/pyosra)

2. [ReactionDataExtractor](https://github.com/edbeard/ChemSchematicResolver)

The easiest way to do this is using **conda build**. Download the recipes in `meta.yaml` files, enter the download directory and type

    conda build .

You might need to build **Pyosra** first - note that compilation can take up to 30 minutes. This will create compressed files (.tar.bz2). Create a conda channel by moving the two files into a single directory named 'linux-64' and after changing to an outer directory type:

    conda index .
This creates a local conda channel, which can be used to install the software with all dependencies by typing:

    conda install -c <path/to/tarballs> reactiondataextractor

Finally, download the data files of o **ChemDataExtractor** - one of its dependencies. This is done with the followin:
    
    cde data download

Congratulations, you've reached the end of the installation process!

# Getting Started

To quickly get started, you can run ReactionDataExtractor on a single **image** in the following way:

Open a python terminal and import the library with: 

    >>> import reactiondataextractor as rde
    
Then run:

    >>> result = rde.extract_image('<path/to/image/file>')
    
to perform the extraction. 

This runs ReactionDataExtractor on the image and returns a ReactionScheme object, which contains all the extracted objects packed inside a graph representation of the reactions scheme. Printing a result of a simple reaction scheme gives the following result (currently only chemical compounds represented as diagrams are extracted, so a by-product would not be shown):

    >>> print(result)
    'Brc1ccccc1 + B(c1ccccc1)(O)O --> c2ccc(c1ccccc1)cc2'

Alternatively, more information on a reaction can be shown using a long_str() method.

    >>> print(result.long_str())
    '[ReactionStep(reactants=(Brc1ccccc1, label: Label(Text: 1), B(c1ccccc1)(O)O, label: Label(Text: 2)),products=(c2ccc(c1ccccc1)cc2, label: Label(Text: f3)),
    ------
    Step conditions:other species : ['Pd(PPh3)4', 'Bu4NBr', 'K2CO3']
    temperature : {'Value': 80.0, 'Units': 'C'}
    yield : {'Value': 90.0, 'Units': '%'}
    ------
    )]' 

The segmented image can also be vieved using a `draw_segmented()` method. The method has an optional 'out' parameter which allows saving the image in the following way:

     >>> output_img = result.draw_segmented(out=True)
     >>> output_img.savefig('output/path')
