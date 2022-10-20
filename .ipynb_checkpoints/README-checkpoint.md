# RoofN3DNet

This repository provides Python code for a Deep Learning based geometric reconstruction of simple 3D building models from airborne laser scanning (ALS) point clouds contained in the RoofN3D dataset, as described in the paper ["3D Reconstruction of Simple Buildings from Point Clouds using Neural Networks with Continuous Convolutions (ConvPoint)"](https://doi.org/10.5194/isprs-archives-XLVIII-4-W4-2022-61-2022).

The implemented neural network combines the part segmentation architecture using continuous convolutions as proposed in [ConvPoint](https://arxiv.org/abs/1904.02375) for point-wise roof face segmentation, and [PointNet](https://arxiv.org/abs/1612.00593) modules to determine the (slope and presence) properties of roof faces. From the 2D bounding boxes and the predicted roof face properties, 3D building models are constructed using half-space modeling, which can then be easily converted into triangle mesh (or polyhedral) models.

![RoofN3DNet Architectur](./doc/fig_architecture.jpg)


## Implementation

The implementation is based on the [ConvPoint](https://arxiv.org/abs/1904.02375) GitHub repository as found at  https://github.com/aboulch/ConvPoint. To improve maintainability und readability, this repository only contains those folders from the original repository that are necessary for RoofN3DNet.

## Dependencies

See the GitHub repository of [ConvPoint](https://arxiv.org/abs/1904.02375) at https://github.com/aboulch/ConvPoint for further information on the dependencies of ConvPoint, and how to install the [NanoFLANN](https://github.com/jlblancoc/nanoflann) nearest neighbor module.

Additional dependencies are:

- Pandas
- SciPy (for halfspace intersection, convex hull)
- Shapely (for wkt, geometric objects)
- e13tools (for NumPy utilities)

(Please note that the list might be incomplete.)

## License

Code is released under dual license depending on applications, research or commercial. Reseach license is GPLv3.
See the [license](LICENSE.md).

## Citation

If you use this code in your research, please consider citing the [ConvPoint](https://arxiv.org/abs/1904.02375) and [PointNet](https://arxiv.org/abs/1612.00593) papers as well as:

```
@Article{isprs-archives-XLVIII-4-W4-2022-61-2022,
author = {Kada, Martin},
title = {3D Reconstruction of Simple Buildings from Point Clouds using Neural Networks with Continuous Convolutions (ConvPoint)},
journal = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
volume = {XLVIII-4/W4-2022},
year = {2022},
pages = {61-66},
url = {https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLVIII-4-W4-2022/61/2022/},
doi = {10.5194/isprs-archives-XLVIII-4-W4-2022-61-2022}
}
```

## RoofN3D Data

For convenience, the repository contains the [RoofN3D](https://roofn3d.gis.tu-berlin.de/) data in the respective folder. See the papers ["RoofN3D: Deep Learning Training Data for 3D Building Reconstruction"](https://doi.org/10.5194/isprs-archives-XLII-2-1191-2018) and ["RoofN3D: A Database for 3D Building Reconstruction with Deep Learning"](https://doi.org/10.14358/PERS.85.6.435) for more information. The data can also be downloaded from https://roofn3d.gis.tu-berlin.de/.

## Training

The RoofN3DNet model can be trained with the RoofN3D data (from within the RoofN3DNet folder) as follows:

```
python RoofN3DNet_Train.py --savedir "../Saves/" --epochs 40
```

The models is trained for the given number of epochs, and the models weights saved in the specified folder.

By default, the data is split into 85% training, 10% validation, and 5% testing. You need to adapt the construction of the RoofN3DDataset object in the RoofN3DNet_Train.py file for other splits.

## Prediction

The [Saves](Saves/) folder already contains pre-trained weights that can be directly used to predict the roof face segmentation and roof face parameters for 625 random buildings that are then constructed in 3D as triangle meshes organized in a (25x25) grid arrangement as shown in the experiments and results section of the paper: 

```
python RoofN3DNet_PredictGrid.py --savedir "../Saves/2022-10-18-18-54-25"
```

The roof face segmentation and constructed 3D models are saved in the [Results](Results/) folder.

Please be aware that the model construction using half-space modeling sometimes fails, which is probably caused by buildings with very low heights. Try to run the prediction again, or try to lower the position of the feasible point in the construct_3Dmodel() function in RoofN3DModel.py. The code line is marked and should be easy to find.

Example outputs are available in the [Results](Results/) folder, and can be viewed, e.g., with CloudCompare. The 3D building models look more appealing if the normal vectors per triangle of the meshes are first computed within CloudCompare, and then a different color than the default is given.

![fig_results.jpg](./doc/fig_results.jpg)

