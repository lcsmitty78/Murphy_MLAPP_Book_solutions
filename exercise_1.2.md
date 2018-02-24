# Approximate KNN classifiers

The aim here is to use FLANN library to find K nearest neighbours.
Well, as of now, I think the task consists from three steps:

1) Install the library, test that it works on a trivial example
2) Understand better the KNN code
3) Do measurements.

Go!

My input parameters for this tasks are:
  * MacOS
  * Gnu Octave 4.2.1
  * cmake 3.7.2

## Install the library, test that it works on a trivial example

So, I started with checking a pre-compiled version for MacOS available at
http://www.pointclouds.org/downloads/macosx.html . It is 1.7, but the
latest available is 1.8.4. So, I will try to build from the source.

Download, unzip, build.

```
$ wget http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann-1.8.4-src.zip
$ unzip flann-1.8.4-src.zip
$ cd flann-1.8.4-src
$ mkdir build
$ cd build
$ cmake ..  # yes, I have some of the development tools installed already
...
-- Building matlab bindings: ON
...
$ ls -lh build/src/matlab/
total 11632
drwxr-xr-x  5 knikitin  593637566   170B 24 Feb 10:58 CMakeFiles
-rw-r--r--  1 knikitin  593637566   7,7K 24 Feb 10:58 Makefile
-rw-r--r--  1 knikitin  593637566   1,7K 24 Feb 10:58 cmake_install.cmake
-rwxr-xr-x  1 knikitin  593637566   5,6M 24 Feb 11:05 nearest_neighbors.mex
-rw-r--r--  1 knikitin  593637566    57K 24 Feb 11:05 nearest_neighbors.o
```

Test.

```
$ cd src/matlab/
$ octave-cli
> test_flann
error: handles to nested functions are not yet supported
error: called from
    test_flann at line 78 column 5
```

So, the problem here is `run_test` in octave does not work.
In general, this function does nothing more than just wraps
another function call. So, for every `run_test` entry we can
replace it buy something like:

Before:

```
run_test("test_load_data", @test_load_data);
```

After:

```
ok = 1;
test_load_data();
fprintf('done test_load_data : %s\n', cell2mat(outcome(ok+1)));
```

After these modifications:

```
$ octave-cli
> test_flann
error: load: unable to find file ./dataset.dat
```

Oh, yes, comment in test_flatt is very specific about that.

```
$ wget http://people.cs.ubc.ca/~mariusm/uploads/FLANN/datasets/dataset.dat
$ wget http://people.cs.ubc.ca/~mariusm/uploads/FLANN/datasets/testset.dat
```

One more attempt:

```
$ octave-cli
> test_flann
done test_load_data : PASSED
error: 'nearest_neighbors' undefined near line 82 column 26
error: called from
    flann_search at line 82 column 24
    test_flann>test_linear_search at line 85 column 22
    test_flann at line 88 column 5
```

I hope it's because of `nearest_neighbors.mex` file is located in the build folder.

```
$ cp ../../build/src/matlab/nearest_neighbors.mex .
```

Trying:

```
octave:1> test_flann
done test_load_data : PASSED
done test_linear_search : PASSED
done test_kdtree_search : PASSED
done test_kmeans_search : PASSED
done test_composite_search : PASSED
done test_autotune_search : FAILED!!!!!!!!!
done test_index_kdtree_search : PASSED
done test_index_kmeans_search : PASSED
done test_index_kmeans_search_gonzales : PASSED
done test_index_kmeans_search_kmeanspp : PASSED
done test_index_composite_search : PASSED
done test_index_autotune_search : PASSED
```

Yay! So, we have some problems with autotune. I don't want to figure out why :)

So, copy `build/src/matlab/nearest_neighbors.mex` and `src/matlab/flann_search.m`
to the `pmtk3/demos` folder. And let's do fun coding!

At first let's try to understand arguments to `flann_search`. Open documentation.
Read it.

The documentation is not precise about arguments to the `flann_search`.
It claims that the first argument should be index, but in the `test_flann`
we saw that there is a way where `flass_search` builds index by itself.

So, let's try from the simple example.

```
$ octave
octave:1> data = [ 1 1 1 1 1; 2 2 2 2 2; 3 3 3 3 3; 4 4 4 4 4]
data =

   1   1   1   1   1
   2   2   2   2   2
   3   3   3   3   3
   4   4   4   4   4

octave:2> test = [ 0 1 -1 2 1; 3 3.5 4 4.5 3.5 ]
test =

   0.00000   1.00000  -1.00000   2.00000   1.00000
   3.00000   3.50000   4.00000   4.50000   3.50000

% Check it using sqDistance. As expected - first vector is closer to [1 1 1 1 1], second to [4 4 4 4 4].
octave:3> sqDistance (data, test)
ans =

    6.0000   37.7500
   15.0000   15.7500
   34.0000    3.7500
   63.0000    1.7500

% Attention! flann_search needs transponed data as an input.
octave:4> flann_search(data', test', 1, struct('algorithm','kdtree','trees',8,'checks',64))
ans =

   1   4
```

Ok, so this is the source code for out MNIST FLANN!

```
loadData('mnistAll');

trainndx = 1:60000; testndx = 1:10000;

% Respape data to be compatible with FLANN

ntrain = length(trainndx);
ntest = length(testndx);
Xtrain = double(reshape(mnist.train_images(:,:,trainndx),28*28,ntrain));
Xtest  = double(reshape(mnist.test_images(:,:,testndx),28*28,ntest));

ytrain = (mnist.train_labels(trainndx));
ytest  = (mnist.test_labels(testndx));
clear mnist trainndx testndx; % save space

tic
nearestIdx = flann_search(Xtrain, Xtest, 1, struct('algorithm', 'kdtree', 'trees', 8, 'checks', 64));

ypred = ytrain(nearestIdx);
errorRate = mean(ypred ~= ytest);
fprintf('Error Rate: %.2f%%\n',100*errorRate);
t = toc; fprintf('Total Time: %.2f seconds\n',t);
```

Run it.

```
Error Rate: 3.39%
Total Time: 6.77 seconds
```

4x speedup (I am sure it could be better if we will play with types of trees
in FLANN and so on) and 10% worse results.