### Changes made to this project 

#### Focus on:
- co-attention
- cross-attention
- learnable parameters
  - adaptive weights
- content aware gating



#### Notes;

##### CO Attention 
1. Self-Attention
2. Add and Norm
3. Feed Forward
2. Co-Attention 
   for each modality:
    - Q from Self Attention 
    - KV as combination of Self Attention output of both other modalities
3. Add and Norm (FFN and Co Attention)

##### Cross Attention 
1. Self-Attention 
2. Add and Norm
3. Feed Forward
4. Cross Attention 
   - Q from Self Attention 
   - KV from the "next" modality
5. Add and Norm (FFN and Cross Attention )
