# Issues to be resolved

### 1. Assigning discriminator parameters with pretrained values

In def train() self.weightsD parameters need to be the same of the weights assigned to the main discriminator parameters self.d_params.
Otherwise there is a mismatch and it will not work. 

Therefore, the LSTM architecture used for the pretrained discriminator should also be applied for the main one. At the moment they
are both different as the main discriminator is a simple feedforward network.

### 2. Animation of the generated distribution

At the moment I need to download the software that will save the images in bitmap form which then can be saved as a video. 
This is useful for visualization purposes.
