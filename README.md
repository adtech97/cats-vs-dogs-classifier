# cats-vs-dogs-classifier
This is simple covolutional model which performs classification of cats and dog once given the image. Model training steps, and information about loss and accuracy is present in Model-stats.ipynb.
Currently Underfiiting and overfitting are not taken care of as project is in very early stage PRs are always welcome (that's how we can learn and grow togather) making this repo specially for people who are new to world of deep learning and computer vision.

## Prerequisites
1.  tensorlfow >=2.X
2.  python >=3.6.X
3.  pandas

## Clone and use 
Git repository can be cloned with git clone also can directly download from git hub

Step 1: Clone repository in local directory
Step 2: setup conda environment (Not compulsary to use conda recommend it because iteasy to setup)
        conda setup guide will be given below
Step 3: Open Model-stats.ipynb run the model.

## Conda env setup guide
1. Download and install Anaconda. For download link   [click here] (https://www.anaconda.com/products/individual)
2. Its recomeded to create seperate conda eviornment 
   for env creation use command conda
   ```bash
   create -n my-env-name
   ```
3. Install tensorflow you will find tensorflow in conda package itself
   ```bash
   conda install tensorflow  -- for cpu version
   conda install tensorflow-gpu -- for GPU version
   ```
4. after step 3 you can run .py files but for jupyter u have config custom kernel.
5. For kernel installation please go through following medium article
  [kernel-installation-guide] (https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084)
   

## Contributing
As i mentioned in description PRs are always welcome but for major changes please open issue first with explaination why we need to change this. Once PR is made upload test reports and comparision stating how your change improves model.

## Dataset used for project
Dataset is available on kaggle u can find it here [dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats/data)

## Credits

[cats-vs-dogs-basic-cnn-tutorial](https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial)
