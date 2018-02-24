As in exercise 1.1 we have some paths missing.

```
addpath('../data/knnClassify3c/')
addpath('../matlabTools/graphics/')
addpath('../toolbox/SupervisedModels/knn/')
addpath('../matlabTools/util')
```

Also in `../matlabTools/graphics/printPmtkFigure.m` we need to set `printFolder`
to something existing.

In `knnClassifyDemo` call to `errorbar` does not work for Octave.

```
error: errorbar: data argument 5 must be numeric
```

I think that octave's `errorbar` does not support some extra arguments like `linewidth` etc.
So I rewritten this as

```
errorbar(ndx, mu, se, 'ko-');
```

Result: nice plots! I hope I will be able to do something like that :)