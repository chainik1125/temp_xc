## Stuff to do for reviewer responses


1. Need to show that the the temporal information is what is important for the prediction.
    - Simplest thing to do seems to be to add the temporal benchmark. 
    - Note that we already show this for clock tasks (monotonically improving accuracy past threshold I think, the "Rayleigh W scan")
    - Maybe also the denoising bench?
    - They wanted stacked SAE, so we can add it - should be easy, if unprincipled.
    
2. If we could show systematic improvement with window length that would be helpful.
    - Han's graph showed something like this?
    - It would also help if we found a regime that did this in the synthetic setting.
3. Need to find 1 to 2 new temporal tasks.

4. If we can improve FreqBench, that would be T-riffic, but maybe not strictly necessary. Again, the crucial thing would be to show (and understand) that we can get systematic improvement with window length. Maybe we can do something like match window length to the task?
    - That's true in all the clock models.

5. We can be cheeky and sell a spectral crosscoder as a 'clarification of the relevant Matryoshka penalty'.
