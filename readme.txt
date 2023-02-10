This project is a re-implementation of the Audio Visual Object Localization net from the paper Objects that Sound by Relja Arandjelovi and Andrew Zisserman.

Joel Huang was responsible for writing the functions to download and process the dataset, the custom dataset class, and visualization code.
Justin Ng wrote the pytorch network models, training and testing functions.
Both parties were heavily involved in the debugging of the entire project when integrating the different parts.

We used Kyuyeon Kim's Github implementation of Objects that Sound https://github.com/kyuyeonpooh/objects-that-sound as reference for the AudioSet downloader, Dataset implementation, and localization visualization. Some details including the data augmentation, dataset cache formats, and data processing methods were taken from their implementation. However, the code generally required heavy modification, and none was copied as-is. Problem Set 7 and 9 were also used as reference for writing the training functions.

