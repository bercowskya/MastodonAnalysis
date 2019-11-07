# MastodonAnalysis
Read and analyse Mastodon (FIJI Plugin) .csv and .xml data

[Mastodon](https://github.com/fiji/TrackMate3) is a large-scale tracking and track-editing framework for large, multi-view images. It allows you to track cells' dynamics over time and has a very useful and easy to use GUI. In order to use Mastodon, since it works with [Big Data Viewer](https://github.com/bigdataviewer), you need your data to be in HDF5/xml format. 

As an output, Mastodon provides either a -mamut.xml or a .csv file which containes, among many features, the XYZ coordinates of each cell and the 3D average intensity of the cells. This package provides the  tools to facilitate the organization of the data and enable the easy creation of figures for spatial, temporal and mitotic dynamics of the cells. 


## Dependencies
numpy \\
matplotlib.pylab
pandas
scipy
xml.etree.ElementTree
untangle



