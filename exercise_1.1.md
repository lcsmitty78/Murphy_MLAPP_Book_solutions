# KNN classifier on shuffled MNIST data

This solution will mostly a description of the installation / configuration process for MacOS.

1. Install `octave`. It is available in [brew](https://brew.sh), so for me it was like `brew install octave`.
It takes some time. As a result:

    ```
    $ octave --version
    GNU Octave, version 4.2.1
    ```

1. Clone `pmtk3` - a repo for this book. `git clone https://github.com/probml/pmtk3.git`.

1. Let's run the example!

    ```
    $ cd pmtk3/
    $ octave
    octave:1> mnist1NNdemo
    error: 'mnist1NNdemo' undefined near line 1 column 1
    ```

    OK, probably we need to `cd` to the folder containing this file.

    ```
    octave:1> cd demos
    octave:2> mnist1NNdemo
    error: 'loadData' undefined near line 11 column 1
    error: called from
        mnist1NNdemo at line 11 column 1
    ```

    Hmm, but wait, `loadData` is something from the different folder. Let's append `find . -name loadData.m` folder to the search path.

    ```
    octave:3> addpath('/Users/knikitin/Development/ml/pmtk3/pmtkTools/dataTools')
    octave:4> mnist1NNdemo
    load: unable to find file mnistAll.mat
    error: called from
        loadData at line 44 column 7
        mnist1NNdemo at line 11 column 1
    ```

    Ok, probably one more folder is missing in path. `find . -name mnistAll.mat`.

    ```
    addpath('/Users/knikitin/Development/ml/pmtk3/bigData/mnistAll')
    octave:5> mnist1NNdemo
    warning: load: '/Users/knikitin/Development/ml/pmtk3/bigData/mnistAll/mnistAll.mat' found by searching load path
    error: 'isOctave' undefined near line 51 column 5
    error: called from
        mnist1NNdemo at line 51 column 1
    ```

    Looks like the code makes an assumption about some variables, that are missing. Let's try:

    ```
    octave:5> isOctave = true
    isOctave = 1
    octave:6> mnist1NNdemo
    warning: load: '/Users/knikitin/Development/ml/pmtk3/bigData/mnistAll/mnistAll.mat' found by searching load path
    error: 'sqDistance' undefined near line 55 column 11
    error: called from
        mnist1NNdemo at line 55 column 9
    ```

    `find . -name '*.m' | xargs grep sqDistance` shows that this function is defined at `./matlabTools/stats/sqDistance.m`.

    ```
    octave:6> addpath('/Users/knikitin/Development/ml/pmtk3/matlabTools/stats')
    octave:7> mnist1NNdemo
    warning: load: '/Users/knikitin/Development/ml/pmtk3/bigData/mnistAll/mnistAll.mat' found by searching load path
    Error Rate: 3.80%
    Total Time: 3.91 seconds
    ```

    Yay! Actually I started to doubt that it is doable. Cool. 
    
    But I don't want to load this paths again. So, we need to run

    ```
    octave:8> savepath 
    warning: savepath: current path saved to ~/.octaverc
    ```

    Then also, append `isOctave`:

    ```
    echo 'isOctave = true' >> ~/.octaverc
    ```

    Ok, let's test it.

    ```
    octave:9> ^C
    pmtk3 $ cd demos
    pmtk3/demos $ octave
    octave:1> mnist1NNdemo
    warning: load: '/Users/knikitin/Development/ml/pmtk3/bigData/mnistAll/mnistAll.mat' found by searching load path
    Error Rate: 3.80%
    Total Time: 3.96 seconds
    ```

    Now back to the exercise.

1. **Run mnist1NNdemo and verify that the misclassification rate (on the first 1000 test cases) of MNIST of a
1-NN classifier is 3.8%. (If you run it all on all 10,000 test cases, the error rate is 3.09%.)**

    So, looks like somehow we ran this sample on 1000 test caess. But we didn't specify anything.
    So, let's open the code editor to understand what's happening.

    Line `#12` is exactly the switch for the size of a test dataset.
    Modifying it to `if 1` and re-running the function:

    ```
    octave:2> mnist1NNdemo
    warning: load: '/Users/knikitin/Development/ml/pmtk3/bigData/mnistAll/mnistAll.mat' found by searching load path
    Error Rate: 3.09%
    Total Time: 28.72 seconds
    ```

    Yeah, actually that makes sense. KNN is cheap in the training stage (just appending new records to the array) and expensive at the application stage (because you need to scan through all samples).
    Because amount of training samples is the same, 60000, complexity should increase linearally.
    We can see something like that. 1000 examples - 3.9 seconds, 10000 examples - 28.7 seconds.
    Definetely loading data consumed some constant time in both cases etc.

    So, I think we are done with this part. Moving on.

1. **Modify the code
so that you first randomly permute the features (columns of the training and test design matrices), as in
shuffledDigitsDemo, and then apply the classifier. Verify that the error rate is not changed.**

    In the `suffledDigitsDemo` there is a code that does it. Probably, that code is

    ```
    if 1
        % to illustrate that shuffling the features does not affect classification performance
        perm  = randperm(28*28);
        mnist.train_images = reshape(mnist.train_images, [28*28 60000]);
        mnist.train_images = mnist.train_images(perm, :);
        mnist.train_images = reshape(mnist.train_images, [28 28 60000]);

        mnist.test_images = reshape(mnist.test_images, [28*28 10000]);
        mnist.test_images = mnist.test_images(perm, :);
        mnist.test_images = reshape(mnist.test_images, [28 28 10000]);
    end
    ```

    Let's check line by line:

    `randperm` - probably, returns a random permutation:

    ```
    octave:4> randperm(10)
    ans =

        3   10    6    8    5    1    2    4    9    7
    ```

    `mnist.train_images`, `mnist.test_images` are probably the names of variables where 
    MNIST data is stored. In the code for `mnist1NNdemo` we are using the same variables. So, OK.

    `reshape` - looks like something to shange the shapes of arrays. Let's check:

    ```
    octave:15> a = [[1 10]; [2 20]; [3 30]; [4 40]; [5 50]; [6 60]]
    a =

        1   10
        2   20
        3   30
        4   40
        5   50
        6   60

    octave:17> b = reshape(a, [3 2 2])
    b =

        ans(:,:,1) =

        1   4
        2   5
        3   6

        ans(:,:,2) =

        10   40
        20   50
        30   60
    ```

    From here it's also obvious, that `()` is used also as an accessor to the array, and you can pass
    there `:` to get a all values from the dimension, a single value and some permutation.

    ```
    octave:19> reshape(b, [6 1 2])(randperm(6),:,:)
    ans =

        ans(:,:,1) =

        2
        3
        4
        5
        6
        1

        ans(:,:,2) =

        20
        30
        40
        50
        60
        10
    ```

    Ok, so the last remining problem are hard-coded values of 60000 and 10000,
    becaue we will be running our tests for 1000 and 10000 test items. So, I put
    the shuffling code after the `ntest = ` expressions. And appended a log line
    to be sure that we are doing reshuffling.

    ```
    ntrain = length(trainndx);
    ntest = length(testndx);

    if 1
        % to illustrate that shuffling the features does not affect classification performance
        perm  = randperm(28*28);
        mnist.train_images = reshape(mnist.train_images, [28*28 ntrain]);
        mnist.train_images = mnist.train_images(perm, :);
        mnist.train_images = reshape(mnist.train_images, [28 28 ntrain]);

        mnist.test_images = reshape(mnist.test_images, [28*28 ntest]);
        mnist.test_images = mnist.test_images(perm, :);
        mnist.test_images = reshape(mnist.test_images, [28 28 ntest]);
        fprintf("Reshuffling was activated\n");
    end

    Xtrain = double(reshape(mnist.train_images(:,:,trainndx),28*28,ntrain)');
    Xtest  = double(reshape(mnist.test_images(:,:,testndx),28*28,ntest)');
    ```

    Run it:

    ```
    octave:20> mnist1NNdemo
    warning: load: '/Users/knikitin/Development/ml/pmtk3/bigData/mnistAll/mnistAll.mat' found by searching load path
    error: reshape: can't reshape 28x28x10000 array to 784x1000 array
    error: called from
        mnist1NNdemo at line 32 column 21
    ```

    Oops, it looks like I was wrong. We need to re-shuffle all data, and then the algoritm takes
    only 1000 first samples to test.

    ```
    ntrain = length(trainndx);
    ntest = length(testndx);

    if 1
        % to illustrate that shuffling the features does not affect classification performance
        perm  = randperm(28*28);
        mnist.train_images = reshape(mnist.train_images, [28*28 60000]);
        mnist.train_images = mnist.train_images(perm, :);
        mnist.train_images = reshape(mnist.train_images, [28 28 60000]);

        mnist.test_images = reshape(mnist.test_images, [28*28 10000]);
        mnist.test_images = mnist.test_images(perm, :);
        mnist.test_images = reshape(mnist.test_images, [28 28 10000]);
        fprintf("Reshuffling was activated\n");
    end

    Xtrain = double(reshape(mnist.train_images(:,:,trainndx),28*28,ntrain)');
    Xtest  = double(reshape(mnist.test_images(:,:,testndx),28*28,ntest)');
    ```

    Run (1000):

    ```
    octave:20> mnist1NNdemo
    warning: load: '/Users/knikitin/Development/ml/pmtk3/bigData/mnistAll/mnistAll.mat' found by searching load path
    Reshuffling was activated
    Error Rate: 3.80%
    Total Time: 3.55 seconds
    ```

    Run (10000):

    ```
    octave:21> mnist1NNdemo
    warning: load: '/Users/knikitin/Development/ml/pmtk3/bigData/mnistAll/mnistAll.mat' found by searching load path
    Reshuffling was activated
    Error Rate: 3.09%
    Total Time: 28.04 seconds
    ```

Done!