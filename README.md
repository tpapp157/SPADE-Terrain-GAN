# SPADE-Terrain-GAN
## Goal
Create a generator model which can translate a human-drawn Segmentation map of terrain into a realistic Texture and Height map which can be rendered in 3D.


## Building the Training Data
Using USGS data, I assembled a global Terrain and Height map. The Terrain map is an artistic depiction of land type and other features as well as geometrically calculated shadowing and highlights to show changes in elevation. The Height map encodes land altitude above sea level as pixel intensity. Five thousand 512x512 patches were randomly taken from the global map with some selection to ensure the crop included meaningful features (not just empty ocean). In addition, the crops were rescaled based on latitude to compensate for the distorting effects of the map projection and ensure a consistent size to terrain features.

To create the Segmentation maps, a dataset was assembled by randomly sampling pixels from the real Terrain/Height maps along with other information about surrounding pixels. Features were reduced using a non-linear dimension reduction and then the samples were clustered. My original intent prior to starting was to use five clusters but after some analysis of the data it became clear that seven would be needed. These clusters were manually tweaked to better align with my qualitative interpretations and finally a model was trained to classify pixels into their appropriate segmentation category. The Terrain crops were processed by this model along with some post-processing to reduce overall noise and produce Segmentation maps composed of large, smooth blobs. Some randomization was introduced into the parameters of the post-processing to increase the diversity of the Segmentation maps and ensure overlapping patches were not identical.

Exmple of Terrain, Height, and Segmentation maps.
![](../master/images/DataExample.png)


## GAN Structure
Unfortunately, the Terrain maps have elevation shadowing baked into their pixel values. This is a problem because this shadowing will conflict with our dynamic shadowing when we render the terrain in 3D. The Generator needs to be trained to produce flat Textures with no shadowing, but I only have access to shadowed Terrain. In theory, the shadowing could be removed from the Terrain maps if I knew the exact algorithm used to create it and so I put some effort into reverse engineering the algorithm with less than perfect results. While my efforts got quite close, they were never perfect and the generator would learn to compensate for the shadowing inaccuracies. After many failed attempts, I gave up on this approach and decided the Generator would need to learn the shadowing algorithm on its own.

My current architecture uses a two step Generator. The core Generator takes the Segmentation map as input and outputs a Texture map and Height map. These are then taken as input by the Shadow Generator which calculates the shadows and highlights based on the Height map and applies them to the Texture map to generate the final Terrain map. This generated Terrain map can then be compared against the true Terrain map by the Discriminator.

One final problem remains however, which is that the shadowing of the true Terrain maps is consistent across the entire dataset (the light source is always shining from the North-West / upper-left). This uniformity means that shadowing information can still leak back into the core Generator resulting in shadows still appearing on the Texture maps. The solution is to create a second, parallel shadowing route in the training architecture. This route takes the generated Texture and Height maps, randomly flips their orientation, and then these new flipped maps are processed by the Shadow Generator and also the Discriminator. By comparing the same maps with different shadowing orientations, the Discriminator can identify and penalize the Generator if any shadow information leaks into the Texture map.

Final Training Flow.
![](../master/images/NetworkFlow.png)


## Generator
The Generator is based on the SPADE architecture with some alterations to better suit my problem. Most notably, random noise was injected into the SPADE Residual Blocks similar to the Style-GAN architecture. This noise significantly helps the Generator to invent features, in particular for large regions of uniform classification in the Segmentation map (the extreme case being a Segmentation map entirely composed of a single classification). Without the random noise, the Generator can collapse to producing repeating patterns in these areas.


## Shadow Generator
The Shadow Generator is composed only of two convolutional layers and the output map is added to the Texture map to produce the Terrain map.


## Discriminator
The Discriminator is a multi-scale, patch Discriminator similar to that used in the SPADE architecture. It takes the concatenated Terrain, Height, and Segmentation maps as inputs and the outputs are used to calculate a Hinge loss. During training, the Discriminator processes the true maps, generated maps, and flipped maps in turn.
In addition, a VGG Perceptual Loss is also calculated to provide an absolute loss term and help stabilize training.


## Training
Many papers have pre-defined training cycles which train the Discriminator more often than the Generator for more stable training. A common technique is to use a set 2:1 D:G training ratio. For my training, I took this a step further and dynamically enabled Generator training based on the Discriminator loss from the previous batch. I defined a threshold range within which I wanted the Discriminator loss to stay. If the loss went above this threshold, Generator training was disabled. If the loss went below this threshold Generator training was re-enabled. Most training epochs averaged out to a training ratio between 1.5:1 and 3:1 (so using a fixed 2:1 ratio is a reasonable choice). However, there were epochs which saw the training ratio go as low as 1:1 and as high as 100+:1 so I do think this method of dynamic training was useful for keeping the networks in relative balance.


## Results
Final results turned out quite well. The detail isn't quite as sharp as in the input data but overall very impressive considering the translation from a fairly vague segmentation map to realistic looking 3D terrain. I also wrote a very simple python function for converting the Texture and Height matrices output by the Generator into an OBJ 3D model file which can be opened and viewed by any 3D rendering application.

Example Terrain, Height, and Texture maps.
![](../master/images/OutputExample.png)

Texture and Height maps converted into an OBJ 3D model file and rendered in 3D (using Windows 10 3D Viewer).
![](../master/images/RenderExample.png)

You can download this [sample OBJ file here](../master/images/3D_Model).
