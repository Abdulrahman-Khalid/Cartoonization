# Cartoonization
Add stickers to your face. School Project.

Face detection using ViolaJones, face segmentation using `dlib` predictor.

![](/scrn.png)

# Install Dependencies:
## first, install python3.7
```bash
$ make install-requirements
```

### Note: if `dlib` didn't work, run 
```bash
$ make install-dlib
```

# To run with our own detection algorithm:
```bash
$ make
```

# To run with `dlib` detection algorithm:
```bash
$ make run-dlib
```

# To run-tests on images
```bash
$ unzip ViolaJones/DataSet.zip
$ make run-tests
```