# deep-generative-statue-inpainting
Masters thesis work

In the field of computer vision, advances have been made in recent years, which have allowed
the creation of systems that, through artificial intelligence, are capable of generating high-
quality images or videos, such as DALL-E, Midjourney, and Stable Diffusion.
On the other hand, new techniques have been employed to facilitate tasks related to
the digitization, reconstruction, and restoration of cultural heritage objects or archaeological
artifacts through computational tools.


In the present thesis work, a mechanism based on deep learning is developed to create
a plausible reconstruction of images of statues with human faces that have suffered some
damage. To achieve this goal, techniques used in other state-of-the-art research, which have
been applied to tasks like facial analysis and the reconstruction of human faces, are utilized
due to their structural and symmetrical similarity with the face of a statue.


During the development of this solution, a collection of images of statue faces extracted
from the web was gathered. These images were processed and subsequently used in the
training of an artificial intelligence model capable of predicting the structure of a statue’s
face with missing information. Another model was trained to reconstruct an image with
missing information while considering the facial structure.


Finally, the performance of the model responsible for reconstructing statue faces was
evaluated under various conditions, varying the amount of information loss in the data used
during its training. In conclusion, the proposed objectives were achieved, resulting in the
creation of a dataset containing statue faces with their respective facial structures, along
with the design of both models. Together, they yield results that are useful for facilitating
the restoration of damaged pieces.
ii



### Fractured Face Statue Landmark Detector on Face Statues with Missing Information

The landmark detector module aims to retrieve a set of 68 coordinates pixel landmarks from a
masked face statue image. Technically, the landmark prediction module can be accomplished
by any landmark detector or regression model that takes some information from an image
and returns the object structure. What we expect from the landmarks detector model is to
obtain information about the underlying topology structure and some attributes like pose
and expression by looking the non masked image regions.

Our proposed landmark detector is designed as a supervised machine learning algorithm
based on the ResNet50 architecture that we will train taking as input a fractured (masked)
face statue image and the ground truth face landmarks predicted by dlib’s library on the
non masked image. Then the model is trained to predict the face structure of fractured face
statues.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12qxhFkJNdX8NbPESIDt_rnJurcsGvUW8?usp=sharing)
[![Static Badge](https://img.shields.io/badge/view_.ipynb-code-blue)](https://github.com/luismontanaresm/deep-generative-statue-inpainting/blob/main/notebooks/detector_de_estructuras.ipynb)




![image](https://github.com/luismontanaresm/deep-generative-statue-inpainting/assets/38935393/96de7eb4-3bec-48ca-aeed-08294bc85f8e)

### Landmark Guided Inpainting on Face Statues with Missing Information

The face statue inpainting network desires to run a face completion to complete masked
face statue images by taking as input masked images and the face structure (ground-truth
or predicted face landmarks). This deep learning neural network consists of a generative
adversarial network composed by a generator model G and a discriminator model D.


The goal of the generator G is to fill pixels of missing parts of masked regions with
appropiate contents by understanding the masked image x and synthesizing the output image
G(x) that preserves texture and face statue attributes from non masked regions, which will
train taking the source images, some random masks and the ground truth or predicted face
landmarks. Then it is supposed to learn to generate plausible results of face statues.
In the other hand, the goal of the discriminator D is to serve as a criticizer that distinguishes
between real (ground truth ) and generated (fake) images. Finally the discriminator will work
as a classifier that receives the ground truth or predicted facial landmarks and generated
images and determine if the image is real or fake.

They are both trained concurrently to improve the quality of synthesized images by the
generator. In this process, when discriminator get better results at distinguishing real images
from fake images, the generator will have to synthesize better image completion to fool the
discriminator.
Figure 4.6: Face statue images with masked regions missing different percentages of facial
pixels

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1vKfxtkLlhh8DV_hajKiqzij3_oZ71d1c/view?usp=sharing)
[![Static Badge](https://img.shields.io/badge/view_.ipynb-code-blue)](https://github.com/luismontanaresm/deep-generative-statue-inpainting/blob/main/notebooks/landmark_guided_statue_inpainting.ipynb)


![image](https://github.com/luismontanaresm/deep-generative-statue-inpainting/assets/38935393/a2d28acb-3d35-4764-a9b8-601003fe827f)
