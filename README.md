# MastodonAnalysis
Read and analyse Mastodon (FIJI Plugin) .csv and .xml data

[Mastodon](https://github.com/fiji/TrackMate3) is a large-scale tracking and track-editing framework for large, multi-view images. It allows you to track cells' dynamics over time and has a very useful and easy to use GUI. In order to use Mastodon, since it works with [Big Data Viewer](https://github.com/bigdataviewer), you need your data to be in HDF5/xml format. 

As an output, Mastodon provides either a -mamut.xml or a .csv file which containes, among many features, the XYZ coordinates of each cell and the 3D average intensity of the cells. This package provides the  tools to facilitate the organization of the data and enable the easy creation of figures for spatial, temporal and mitotic dynamics of the cells. 

## [1] Conversion to HDF5 and XML
Before using Mastodon, you need to convert your files in a format that BigData viewer can read. For this, using either [Big Data Viewer](https://github.com/bigdataviewer), [BigStitcher](https://imagej.net/BigStitcher) or [Multiview Reconstruction](https://imagej.net/Multiview-Reconstruction) from Fiji, you can convert your data into HDF5 and XML. HDF5 will save the raw data whereas the XML file will save the metadata and any transformation performed to the raw data. 

## [2] Using Mastodon
[Mastodon](https://github.com/fiji/TrackMate3) is a very user-friendly Tracking plugin from Fiji. It allows interactive visualization and navigation of large images thanks to the BigDataViewer. Any file that can be opened in the BigDataViewer will work in Mastodon (BDV HDF5 file format, KLB, Keller-Lab Blocks file format, N5 file format, ...). 

With Mastodon you will be able to track large amount of cells in a manual, semi-automatic or automatic way. The outputs from the tracking are two .csv files: name-edges.csv and name-vertices.csv . The first one contains the information obtained from the spots: mean, median and standard deviation of intensity of all the channels; x, y, z coordinates of the centroid of the spots; spots radius; detection quality for each spot; tags and sub-tags for the spots; the individual ID for each spot; the track ID to which each spot corresponds. 

## [3] Using MastodonAnalysis.py to analyze Mastodon data
MastodonAnalysis.py is a collection of classes that allows you to obtain tidy arrays with the tracks' and spots' features in an easy way. 

Below you can find a description for each of these classes:

### Class ```xml_features```:
Gets as input the .xml file from the initial conversion using either BigdataViewer, Bigstitcher or Multiview reconstruction to convert the files into HDF5/XML. 
Returns:
* channels
* dimensions
* width
* height
* number of slices
* x,y,z pixel size
* coordinate units ($ \mu $ m, mm, etc.)

### Class ```csv_features```:

### Class ```ordering_tracks```:

### Class ```xml_reader```:

### Class ```peak_detection```:

### Class ```bulk_peak_analysis```:

## Dependencies
numpy 

matplotlib.pylab

pandas

scipy

xml.etree.ElementTree

untangle



