# Image Classification
# Using Keras’ Pre-trained Model for Feature Extraction in Image Clustering

# Work Flow
# - Use the VGG16 Keras' Pre-trained model to extract relevant features of the images
# - Look for the optimal number of clusters using what's know as the "Elbow Curve"
# - Use KMeans in Scikit-Learn to cluster the set of hotel images based on their corresponding features


#Importing the relevant libraries
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os, shutil, glob, os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = True

model = VGG16(weights='imagenet', include_top=False)
model.summary()


#Declaring the path of the images
imdir = '/Users/arieh/Desktop/Kolleno/HotelImages'
#Declaring where to send the clustering output
targetdir = "/Users/arieh/Desktop/Kolleno/Result/"


#Looping over the images to get the relevant features
filelist = glob.glob(os.path.join(imdir, '*.jpg'))                #filelist holds the images
featurelist = []                                                  #this will hold the features list
for i, imagepath in enumerate(filelist):                          #looping over images
    print("    Status: %s / %s" %(i+1, len(filelist)), end="\r")  #printing the progress for visual purposes
    img = image.load_img(imagepath, target_size=(500,500))        #reducing image size for computational effectivity
    img_data = image.img_to_array(img)                            #transforming the image into an array
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    features = np.array(model.predict(img_data))   #extracting features with VGG16 model
    featurelist.append(features.flatten())         #putting features into the feature list


# Now that we have the features let's look for the optimal number of clusters
# First. What is this optimal number of clusters? What are we looking for here?
# Basically, when clustering (in this case), we can go from having one big cluster that hold all 29 images to having 29 clusters holding one image each. Of course none of these options would be considerend clustering at all, it just makes no sense.
#
# So essentially we are looking for the minimum number of clusters where there is enough difference between them.
# To do this I will use the "Elbow curve" where we plot the explained variation as a function of the number of clusters. Ideally, we should see a strong inflection point where our model fits best.


Nc = range(1, 30)                             #'Nc' represent the number of clusters
kmeans = [KMeans(n_clusters=i) for i in Nc]   #we run KMeans through all possible nº of clsuters

#Now we will calculate the score (variation) betweeen clusters for all possible KMeans calculated above
score = [kmeans[i].fit(featurelist).score(featurelist) for i in range(len(kmeans))]

plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.xticks(Nc)
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.grid()

plt.show()


# Unfortunatelly, we cannot distinguish here a clear point of inflection. At least, we can see that when having the maximum number of clusters (29) the score is zero, which makes total sense.
# However, we already know for sure that we do not want more than, let's say, 10 clusters. So let's run it again for a maximum of 10 clusters to get a closer look.


Nc = range(1, 11)                             #where 'Nc' represent the number of clusters
kmeans = [KMeans(n_clusters=i) for i in Nc]   #we run KMeans through all possible nº of clsuters

#Now we will calculate the score (variation) betweeen clusters for all possible KMeans calculated above
score = [kmeans[i].fit(featurelist).score(featurelist) for i in range(len(kmeans))]

plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.xticks(Nc)
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.grid()

plt.show()


# Again, there is no clear point of inflection. Therefore, I decided to do trial-error (which is not ideal) and I found the most accurate results to be 7 clusters.

# Clustering
# Now that we have our optimal number, let's run our KMeans clustering algorithm

kmeans = KMeans(n_clusters=7, random_state=10).fit(np.array(featurelist))


# Now I will save the result in my laptop so I can visualuze the difference between clusters.
# I will name them according to their cluster. So basically: "cluster_imageNumber"

try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    print("    Copy: %s / %s" %(i+1, len(kmeans.labels_)), end="\r")
    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")
