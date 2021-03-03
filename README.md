# norm\_autocorr

Normalized autocorrelation of complex values

Normalized auto-correlation of complex values
```
         +--->CompMagSquared---->FloatSlidingWindow------------+
         |                                                     |
Samples--+--->CompMulti--->CompSlidingWindow--->CompMag--->FloatDivide--->
         |                          |                                                                                                          
         +--->Delay16--->CompConj---+
```

Delay16 prepends 16 zeros and trims 16 samples from the end
FloatSlidingWindow has a 64 sample window sliding over 4000 samples
CompSlidingWindow has a 48 sample window sliding over 4000 samples 

Max Samples
540 * 80 = 43200 samples
43200/4000 = 10.8

Max Samples is arbitrary. Let's do 550 * 80
44000 samples
44000/4000 = 11

11 times to iterate over 44000 samples
For now don't worry about the max samples.


