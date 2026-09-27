# Notes for writing the paper

## Idea dump




## Narrative idea dump



## Paper components

1. 
### 1. Key claims:

   a. Temporal XC find global features
   This is vague. What does 'global' mean? You could use the language the HMM paper uses.
   b. They allow superior steering on tasks that people care about.
      Unfortunately, this doesn't seem to be true on any of the tasks we care about.  
   
   c. We propose a synthetic benchmark for measuring temporal structure.
      d. Would be nice to supplement this with a causal steering benchmark!

   e. Outperform on Sparse Probing

   f. Moreover, extract different kinds of information, so we can use them in conjunction with existing techniques (but note, you can only really make this point if you also understand _when_ to use these techniques)

### 2. Motivation: Why does this matter? Who cares about temporal features?
   a. In general, I think its not clear if we should be presenting this as: TXCs beat everything (everything temporal?) or if we should be presenting this as something worthy of being at a table of equals ("Another arrow in the quiver"). The latter is obviously weaker, but if we do it, then we have to also address _where_ it would make sense to use the txc. Both relative to other temporal architectures, and relative to conventional architectures.

   b. It would help to have a really clean description of what a canonical example of a temporal feature might be. I think the TFA and T-SAE papers would be good to read for that.

   c. 

### Specific experiments




### Questions to consult the literature on

1. Canonical examples of temporal features
  - TFA intro
  - TSAE intro


### Structure

In the ideal case, I think it would be:

1. Intro
2. Synthetic setting
    - In the ideal case, you would be able to say: "Here are the kinds of features we learn, and here is what they tell us"
        - **TOREAD**: Anthropic TMS paper, see how they told the story there.
        - It would be good if we could identify tuning knobs that make the TXCs better: longer temporal window, regularization, activation func, additional loss terms, hookpoints?
        - That's true, but even being able to benchmark different proposals for temporal architectures to a source of ground truth in the temporal setting is helpful. An important point of the paper is we're putting the field onto a more orderly basis. **Honestly, this point alone could be a solid paper.** Worth writing this paper if only for this.
            - **TOREAD**: Sparse but wrong for figuring out a good bridge between synthetic and real LLM data.
        - Interesting follow-up, probably for later: Can we analytically prove some results about temporal architectures?
3. Qualitative analysis of features. (T-SAE and TFA might be helpful here)
4. Benchmarking on sparse probing. 
  - What is the broad category here? I think the general thing we're doing is 'macroscopic benchmarks' vs. the zoomed in thing we're doing for case studies in the last section.
  - Why not other tasks? I guess T-SAE paper only did sparse probing but they managed to tie it into their overall narrative which helped them get away with it.
  - The equivalent for us would be if we had a way of understanding what crosscoding gives you. Can we easily understand what crosscoding buys you that T-SAE doesn't? Off-the-cuff, it seems it would be mid-scale variance - features that vary quickly but not on single tokens. This is not quite what we want. We want to have access to the long-range variance (for that we need the contrastive loss?), but also the information of what persists. I guess fundamentally it would be interesting to know what is known in the literature about why a feature would persist across a small (O(10)) temporal window, and how that information helps an autoencoder.
  - I guess one interesting way to frame it could be to think about the 'building blocks' of a temporal architecture, and show that you can put all of the building blocks into the TXC but the T-SAE and TFA are both missing some.
5. Case studies
    - Worth noting that the best thing here would be to have a methodological innovation where we have a benchmark which is a panel of model organisms.
    - Emergent misalignment:
        - Why?
        - Current state: Tie
    - Sleeper:
        - Why? Information is temporal, the model must retain the sleeper.
        - Current state? Outperform in ln1, no in others?
    - Backtracking:
        - Why? Temporally present, had to offset to see it.
        - Current state? Outperformed by T-SAE, Goodfire SAEs best
        - I think an important thing here is to note that the fraction of words that are "Wait" is a terrible metric for backtracking behaviour. Would be much better to identify a reasoning dataset and show that the model does better on that dataset in those cases where steering successfully induces a 'wait'.
    - Venhoff base-->reasoning recovery
        - Aniket failed at this but could try again. Its too ambitious I think because steering vecs are too heavy. But we could nonetheless try to do a downsized version of this, or construct a lightweight version of that dataset.

    - RLHF dataset the temp SAE paper uses.

    - Eval awareness?
    - Sandbagging?
    - What does AxBench do? Is this already the steering benchmark?




## Ideas

1. Interesting thought: Can you do the hybrid intuition and steer on steering vectors derived from both methods? That would be very interesting. A good pre-cursor to this would be to see if the TXC archs, the T-SAE archs, and the SAE archs steer well on the same examples.

2. 










