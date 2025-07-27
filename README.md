# PEEK-Var
exploring the application of pruning methods via PEEK variance to vision transformers

based on prior research from NETS lab

# some preliminary results:

the heat map represents the value of the average attention entropy for a specific patch in a specific encoder layer. 
this means that if the value is higher, then that specific image patch is paying attention to many other patches in the image.
if the value is lower, then that specific image patch is paying attention to very specific patches in the image. 

another observation is that the attention entropy value decreases as the depth in the neural network increases.

<img width="800" height="2400" alt="image" src="https://github.com/user-attachments/assets/77adbc1d-03bc-45d4-a65c-9d0209811137" />

i have also created histograms of the attention entropy distribution for each layer in a small encoder vision transformer. i have 
noticed that the distribution tends to be come bimodal in the last (deepest) layer.

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/dc45c4aa-0f8b-4b4d-8b31-0275b07a56ad" />

things to investigate (in no particular order):
- analyzing results using a larger pretrained model
- is the bimodal distribution consistent in the last layer?
- can we learn anything by applying this to language instead?
- what if we analyze the attention entropy distribution of individual heads within layers?
