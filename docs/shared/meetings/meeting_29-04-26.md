## Meeting link:

https://fathom.video/calls/656165062



## Meeting transcript
Impromptu Google Meet Meeting - April 29
VIEW RECORDING - 44 mins (No highlights): https://fathom.video/share/M_sk4wzaSA3wJmJiaXyewMasVj7pgp72

---

0:03 - Aniket Deshpande
  Yeah, but it's like a pretty narrow end, the green bar with the stripes.

0:12 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, so this is what's up, Han? Hey. Yeah, so we were just talking about Aniket's latest thing, so just help me interpret this plot.  So you're saying that the baseline here is saying that the top KSAE and the TSAE perfectly auto-encode, but the held out is like, do they code forward?  Is that the idea?

0:53 - Aniket Deshpande
  Yeah, so it's like a mass thing, so basically it's the solid bar. The Token architecture is obviously reconstructed better than the Temple of CrossCoder just because the architecture is a per position objective.  But then the Temple of CrossCoder, the win that I guess we saw is that it's for when you hold out one token and you try to reconstruct it with the from unmasked, like the other, it's neighbors that are unmasked.

1:34 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Where is the held out token? Is it in the window or is it, sorry, is it before the last token in the window or after?

1:44 - Aniket Deshpande
  It is the last token. So it's... So the window is like the five tokens immediately before the one you're masking?

1:55 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yes.

1:56 - Aniket Deshpande
  Yeah. Okay. Okay.

1:58 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  That's good. That's good. Because otherwise I would have... Yeah, like some information leaks, some information of like the token in the middle would leak to the tokens that are subsequent, but yeah, if it's like the one...  see. That's good, yeah. So what is this, what is this eval? Sorry, where did this come from? This is Tiny Stories, I believe.

2:25 - Aniket Deshpande
  This was just Han passed me, it was like a briefing for an agent for Cloud Code, and I just kind of let it run for a couple hours.  You might have a better, like on the actual setup, because I think you ran other experiments with the setup, but that's how I understand it so far.

2:47 - Han
  Yes, so I've been using the same setup, but I haven't yet like dug into the results. Like my agent ran for 24 hours and I have to read through everything that...

3:01 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, it's not easy, this managing business. You think you're going to save so much time, but in reality, it's a mixed picture.  What is the very negative fraction of variance explained in the H8? Does that just mean that the thing is basically not working at all?  That's what I'm assuming.

3:30 - Aniket Deshpande
  I'm thinking that, I'm guessing it's just that the subsequent H8 is just, yeah, it has basically worse. Because I guess the fraction of variance explained is, it's not standardized for negative values.  It's just 1 minus the MSE over the variance. So then, I guess it should be.

3:58 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Or no, it's not then.

3:59 - Aniket Deshpande
  Yeah, it's not. I'm guessing it's just a really bad reconstruction, so then when you do the subtraction, you get a negative number.

4:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I'm not fully sure why that one is so much worse.

4:13 - Aniket Deshpande
  It's worse than just using the mean, right?

4:19 - Han
  What are you predicting?

4:24 - Aniket Deshpande
  The reconstruction, and it's the mean squared error of that divided by the variance, and then one minus all of that.

4:36 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  It's the mean squared error between some probe trained on the window of activations for the TXC relative to the residual stream activation on the next token, or is it something else?  It's...

5:00 - Aniket Deshpande
  The main-squared error of the reconstruction of the token that's held out.

5:07 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But what are you reconstructing Sorry?

5:12 - Aniket Deshpande
  What are you reconstructing from? The four tokens before, at least that's what I believe it's doing. I'm just a bit confused.

5:28 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think I'm just being a little bit slow. So you're saying you're reconstructing in the same way that you do in the SAE forward pass, is that right?

5:39 - Aniket Deshpande
  Yes. Yes.

5:44 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  What does it mean to mask one versus have no mask?

5:53 - Aniket Deshpande
  I believe it's the way it's held out as in like it's not included when you try to reconstruct from that window.  So if you're from the T equals five window, you're trying to reconstruct, like, do the audio encoding, but without the fifth one, and you're trying to see if that you can, if the MSE is measuring the reconstruction on the fifth one based on leaving it out, so based on, like, ideally getting information from the other four.  But I guess, okay, so I guess let's just start simple.

6:33 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  For the SAE, let's just understand the baseline case. So for the baseline case and the SAE, what is the input and what is the output?

6:52 - Aniket Deshpande
  It's the, the SAE is, well, the SAE baseline case is just a normal. So autoencoding thing where you give it a tokens residual, and then it should be able to reconstruct.  So that's what we expect it to be effectively perfect, but then for...

7:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So this is saying that the SAE perfectly constructs single token activations?

7:19 - Aniket Deshpande
  Yeah, basically, which is why the held out version is zero, because if you hold out the token, it can't reconstruct it.  It's just kind of like a very simple case.

7:30 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But then I'm confused, right? Because if you're holding out the same token that you're reconstructing, what is the input to the SAE?

7:54 - Aniket Deshpande
  I'm not sure.

7:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, maybe I should dig into what the agent said. So assuming the agent has done something reasonable, which it's 50-50 whether it knows better than me what's reasonable, there is some potential interesting advantage that the TXC has.

8:18 - Aniket Deshpande
  Yeah, okay. I can look into what the SAE case is because I did not think of that. Yeah, it's helpful to understand these things fully.

8:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I just haven't had a chance to look through, so I don't really know. Let me just summarize where I think we're at quickly and say a little bit more about some of the results I got over the weekend.  So what I was meant to do by today was finish a skeleton draft of the paper. I have not done that because I was trying to hunt for other case studies because I'm worried about what this implies for the story that we have.  So I ran a bunch of things, some of them more interesting than others. So the tiny story stuff is the one which maybe has some concrete follow-ups.  So I'm rerunning some seed averaging here. But basically what I'm doing is at each hook point, I'm training an SAE of some architecture.  And what I'm asking is, can I suppress the sleeper behavior in tiny stories? So I'm training this on the sleeper model, sleeper data variant of the SAE variant.  And I'm asking, can I suppress the sleeper behavior by steering on some feature from the SAE? So similar set up to some of the other steering experiments we've been doing.  And the interesting thing here is... Both the TSAE and the Temporal Cross Coder do the best relative to the SAE when you train them at the hook points that are around the tension.  So LN1 is immediately before a tension. I don't have it here, but a tension out immediately after we get similarly good effects.  Now, this is not amazing because if I look at the SAE and Resid post, it also gets this perfect suppression.  And so what I would like to say is that if you train the TXC at the right hook points, it will outperform the SAE at any hook point, not what we see here.  But the interesting takeaway is that the TXC and the TSAE perform much better in these attention hook points. So...  So... Then they do if you train them on the residual stream. That's kind of interesting because most of the temporal information has to be mediated through these hookpoints because it has to be read in by attention.  I had hoped that the TXC would be powerful enough to sort of like disentangle these two things within the residual stream, but it may be the case that those are the hookpoints that we should try.  And so I'm testing some variants of this in the misalignment setting doing that. So I think the takeaway for us is this naive case study seems to suggest maybe training at the attention hookpoints might be an interesting thing to do.  It also might be interesting to train a transcoder variant which is reading in from LN1 and writing or decoding to attention out.  And the idea that... There is, like, you get the nice transcoder intuition of, like, you're capturing the computation, therefore the features you find should be more robust, and in theory, attention out is the right hook point to look at if what you want to do is steer, because that's what we're going to write to.  So, yeah, this case is, in a sense, a really good case because the steering coherence tradeoff is extremely clean.  In a sense, it's a bad case because the SAE already saturates this benchmark, so the fact that the suppression of the sleeper is perfect in ResidPost is already telling you it's going to be hard to be perfect in this benchmark.  Let me just pause there and see if anyone has any comments or questions.

12:49 - Aniket Deshpande
  I guess I took, like, your results just for the Jake Ward paper that I'm trying to reproduce. I was able to reproduce his paper, like his numbers exactly, and now I'm trying it.  with a temporal cross-coder. So basically, we could run a reasoning model's traces through a temporal cross-coder and then try to see if the same backtracking feature fires at the same offsets in the reasoning model.  And the hookpoints I'm using are residual pre, which is the one he uses in his paper, so I just wanted to use the same one there.  Pre-attention, so I used LN1, I'm using 1.1, I believe, which was the best result from your table, so I wanted to use that one.

13:41 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Could you say that last bit again?

13:44 - Aniket Deshpande
  Yeah, I'm just reusing the hookpoint that you had your best results on, which was layer norm 1, I believe.

13:51 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay, okay.

13:52 - Aniket Deshpande
  And then I'm also using Attention Out just to see what happens after the mixing is computed. Yeah, I mean-

14:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  If we had more time, there'd probably be some interesting theory that we could do to try and understand a priori, can we understand which is the best hookpoint, but since we don't have any time, I think we just trade time for compute and just train on all three and see what happens.  I should, by the way, be able to get us some more compute, so bear with me, but hopefully by tomorrow or in a couple of days, I should be able to get us enough compute to just run a bunch of agents.  Which would be helpful. Yeah, this backtracking stuff, I think I'm more interested in than the, like, Wenhoff reasoning-style vectors.  I think that case is just too complicated, and I don't think we really, like, I'd be optimistic about getting it to work eventually, but that's not, I don't think that's what we should focus on now, because it's just too complicated.  So I'm optimistic about this backtracking stuff. The fact that they have these offset positions makes me think... So that this actually is something that's really good for the temporal cross-coder architecture.  So I'm super curious to see what happens. I gotta be honest here, I'm not super optimistic, but I'm like, let's do the best we can and just try and understand the results to the best of our ability.  Yeah. Let me say something else. Oh, yeah. Yeah. Yeah. Sorry. I slightly misspoke in this sentence here. So in this one, I just, you know, I think it's important for all of us to take a step back.  And aside from the paper, and the deadline and all that stuff, just think about What is the actual reason we have for looking into temporal cross-coders in the first place?  And the way I would summarize it is, the initial reason we got excited is in the synthetic setting, it seemed like they were doing a good job at global feature recovery, better than TFA, which was the big update for me.  We also then saw that they have some advantage in sparse probing, and so it seems like that transfers over to language models.  And then it seemed like there were some case studies which prima facie they were looking good on. So I think it's important to reassess all of those lines of evidence.  And so I did that. In the synthetic setting, let me see if I can find the graphs for this because they're kind of interesting.  Oh yeah, I think they're here. I think they're Here. And I think this also motivates a line of work.  Let me just figure out how to share this. Sorry, what am doing? What am I doing? I'm sharing my entire screen.  Can you see my screen? Yeah. So I basically redid Han's early experiment that looked at global versus local features, but I added the temporal SAE to this.  And the results are really interesting. I think this graph is the most interesting for me. So if I look at the area under the curve for global feature recovery, unsurprisingly, the stacked SAE is always the worst.  But then the temporal SAE... The is worse than the temporal cross-coders for low-K, but the temporal cross-coders collapse for high-K, whereas the temporal SAE maintains pretty good global future recovery as we increase-K.  And to me, the takeaway from this is, I think it's similar to what we've talked about before, but I'll say it again.  It feels to me like we're under-regularizing our temporal windows, and this is causing some of these weird results. And a potential direction that this suggests, which, you know, maybe has a high enough probability of success that it justifies agent effort, but maybe not so high that it justifies human effort, is I think there is a framing of what we're doing.  everyone. Which is, we are doing the cross-coding generalization of temporal SAEs, in which case the question becomes, what is the right way to implement the contrastive loss of the temporal SAEs, along with the effective matrioshka penalty that they use, in the cross-coding setting.  And I think one thing that would be really interesting is to say, get an agent to say, okay, I'm going to match the, I'm going to start off by matching the temporal SAE with its contrastive loss, and then as I grow the temporal window, I'm basically going to try and like piggyback where, I grow the temporal window to length two, I keep the same contrastive loss.  If the, um, global feature recovery goes down, then I increase my contrastive loss or change it. In order to try and restore it, and I see if there's some way in which I can choose a contrastive loss such that it continues to reproduce good feature recovery as I increase K in the temporal cross-coding setting.  So, yeah, I think that's worth bearing in mind as one of the ideas for how we could do this thing, which I think is afflicting our TXCs, which is that I think they're just under-regularized.  Any questions?

20:43 - Han
  So, when I was still doing the hail climbing on the Instruction 2 model, I think the last thing I found was if we just picked a really long...  contrastive loss-like distance. So not one or two positions, but like 10 positions or 20 positions. I think that actually helped a lot.

21:09 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I am interested in what if we just do the dumb thing of setting the contrastive window equal to the temporal window, right?  Or at least order of, right? Yeah. So I think there's still a lot of hill climbing to do.

21:24 - Han
  I think in the last couple of days, I diverted all of my computer, like reproducing it on the base model.  Yeah, So I guess, like if you look at the, I guess with the T, the window size sweeps, we have some reason to believe that my hill climb models scale better as we grow T.  But I haven't, I have not concluded that investigation. So it's... Bear with me just one second.

21:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah. I think... think...

22:00 - Han
  Let me give a better plot, like a zoomed-in plot. This is not conclusive, and this is on a subset of all the tasks to make my probing results faster, so it's like half the tasks.  And I find that, okay, firstly, this hill-climb model seems to be above the bare-bones temporal cross-coder, but I think it's also highly dependent on which task we look at.  I think we could also make a case that there are some tasks where it's just much better. I need to look at a per-task breakdown.  I think that would be useful, like a per-task temporal window scaling breakdown. So from my current numbers, I feel like there's huge variation in the ranking if you just look at certain tasks.  I need to get a clear picture on that as well.

23:12 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So that's interesting. Let me just make sure I understand what you're saying. Are you saying, depending on the loss penalty you choose, you get a different distribution for where the temporal cross-coder does well versus where it does badly?

23:25 - Han
  Yeah, so that's what I see so far. You have some choices of regularization where it makes it really good on certain tasks and worse than others.  So I think the regularization might also be task-dependent.

23:44 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Interesting.

23:45 - Han
  It's not something that you can just... I think it's like, okay, maybe there is a global parameter that works well across the board, but so far, Devim points to like, maybe there are some tasks where you just need to treat.  which is the temporal information differently than other tasks, and I don't have a strong theory of why that is the case, and the other thing is I think the sparsity, so the choice of k matters, and it's underexplored, I think, because I've been fixing k, like the total k in the window to 500 for all of my temporal crosscoders, and I never really looked outside that.  I have some kind of evidence that we have like, give me a sec, let me just remind myself of where the plot is, so when I was looking at the temporal SAE, there are two versions, there's one with k equals 20, which is the exact thing that they use in their paper, and then that is like the lowest, the high,  And then there's one with K equals 500 to match my temporal cross-coders, and the K equals 20 one does much better on the steering than the K equals 500 one from...

25:16 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  In their ROHF task.

25:20 - Han
  Yeah, in their ROHF task. So the stupid thing that I didn't try is what if we did temporal cross-coders at K equals 20 or something smaller.

25:31 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I agree. I agree. There is maybe... I feel like we're... It's very understandable, but there is maybe a fair point to be made that we're trying to do these gigabrain hill-climbing of like, oh, what if you add this sub-window sub-sampling or this contrastive logs across the whole window?  And I think those are interesting ideas, but yeah, maybe there's something to be said. Why don't you change K and see what happens?  Probably a reviewer would ask those questions. I think it's like changing both K and the temporal window and seeing how the two of those scale with respect to each other is probably pretty important.  Yeah.

26:27 - Han
  And I think in order for me to get results in time, I'll reduce my 36 task set to a smaller task set because I feel like having 36 tasks isn't that helpful.  And it's like there are some tasks which are kind of repeated many times in like, you know, this task version one, version two, version three, and it's not really that informative.  So, um, yeah, I agree.

26:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think the, like the more we can like scrap, If test some stuff, the better signal I think we're going to get.

27:06 - Han
  Yeah, so I think my current plan is to reduce my task set, get a final set of sparse probing results on that reduced task set, and then add more variants, add different cave values of temporal cross-coded.

27:32 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I just want to be a bit cautious. I can sense myself getting excited about that, but I guess the reason not to do this is right now our sparse probing benchmark is fine.  It's not amazing and I'm sure we could do better, but it's good enough. For us to have a coherent paper, if we don't have a single case study where we can understand why a temporal cross-coder is doing something useful, I think that basically takes the paper that we have from something that I think is really strong and the field should really know about to like, this is kind of a mild curiosity and I suspect it probably won't get accepted in Europe, which is fine.  That's not the metric that we like ultimately care about, but if I'm being honest about like the finite resources of time, hopefully less so now and compute that we have, I would think that the thing that really matters is finding some case study and so maybe it's difficult because I think the point of doing the sparse probing is more instrumental than we really want a good sparse  I think the point of doing the sparse probing is that so we can learn what the right temporal architecture is.  But maybe what I would advocate for is let's do that hill climbing, but across a panel of these tasks that we're investigating.  So like the backtracking, the emergent misalignment, the tiny stories, and I think maybe we should focus there. And probably maybe it's like, I would, I guess let me put it this way, I would feel much more optimistic about us hill climbing on some like behavioral task and transferring that to the sparse probing than hill climbing on the sparse probing and transferring that to some behavioral task.  Yeah, I see.

29:45 - Han
  Okay, yeah, I guess, which, what would be the two most, I don't know, prioritized case studies for this hill climbing?

29:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, so I think what Aniket is What with backtracking is probably currently the most important one. I think emergent misalignment is interesting.  I've got a lot of negative results for emergent misalignment, so I'm kind of losing hope, but it might be worth trying that one because there are some theoretical reasons to think it should work in that case, although it's a little confusing.  I think the tiny stories one is interesting as like a let's conceptually understand what's going on and see if we can hill climb.  The one problem with it is it's just saturated. So like I think if we could have a task which is like toy task, we just want to like de-confuse ourselves by thinking about that toy task.  And if we could have two or three which are like interesting task, big if true if the model like does well on this, therefore we should zoom in on those parameters.  So definitely one of the big if true. The tiny stories is okay as the deconfusion task, but it's saturated, so it's a big problem.  The other tasks, like emergent misalignment is okay as a big if true task, but I'm not sure... I'm growing less confident that my intuition for why it should work is as good as I thought it was.  The ROHF steering that is in the TSAE thing is a reasonable thing to do just because they did it.  They showed that we can find... I guess maybe I would say this. One thing that might be interesting is to try and do this piggyback hill climbing from the TSAE in their steering task.

31:50 - Han
  Yeah, because I think that one is quite easy to run compared to other things. Yeah, so I guess I could start there.  Like, hill climbing by changing...

32:03 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So what I would be excited to try first in the TSAE case is just do this piggybacking of let's start from this TSAE, let's start from the degenerate case of the cross-coder has size 2, window size 2, and uses the same contrastive loss as they do, and let's compare what the steering looks like there.  If the steering is much worse on that case than the TSAE, that's already very interesting, and I think we should vary both the contrastive window and the, what's its face, and the K, in order to see if we, sorry, the contrastive window, the K, and the implicit Matryoshka penalty, because the...  The TSAE does have this implicit penalty, so if we vary those things, and at each point it's like, once we outperform where we were before, we grow the temporal window, so then we go to T equals 3, or T equals 5 if want to move in higher jump sizes, vary those three parameters again so we can outperform, and we keep trying to piggyback in this way.  And the concrete metric that we can use is just the area under the curve for the coherent suppression graph.  Okay, yeah.

33:37 - Han
  So starting from the temporal SAE and trying to make it like a cross-coder, gradually. Exactly, exactly.

33:49 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And we just want to like piggyback the temporal windows across the three parameters on which I expect things could vary.  The K, the contrastive window size... and the Matryoshka Penalty.

34:02 - Han
  Okay, I think we can apply the same idea to another case study where Temporized AEs are good. Or maybe for some diversity, for the other case study, can start piggybacking from another starting point.

34:20 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, we could just take, for the regular, for the backtracking and for emergent misalignment actually, one thing we could do is just try and do this piggybacking from a regular SAE.  So just pick a regular cross-coder, pick a regular SAE, see what the SAE's results are, and then just try and again do this piggybacking where we vary K, we vary some other things, I'm not sure what the relevant parameters are in like the regular SAE comparison.  But yeah, and we try and do this piggybacking to grow the temporal window.

35:00 - Aniket Deshpande
  Would that mean, for the backtracking example, that mean you have to retrain at every cell on this grid sweep that you're making of K?

35:13 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, that's what makes it expensive. So I think what we can do, what we should do maybe, is we should define some scrappy research parameters.  So, you know, train the SAE to like 5k steps or whatever. So maybe the thing to do is like, when we start this, we train an SAE for a long time.  We try and see how well we can do on the backtracking case. And so long as it's reasonably close to how well you can do with difference of mean steering, then we can say, okay, how short a training time can I get away with?  Right. And if we, you know, ideally we'd see some like step. Where it's like, okay, it doesn't matter for a long time, then it really starts to matter.  We may not see that, so we may just have to choose a cutoff, but we just choose an experimentation parameter, and then we choose a scaling parameter, and then I think the way it should go is for each of these piggybackings, once you find in the low training time regime, a regime where you're happy that the TXE is outperforming the previous temporal window, then you should replicate by training for longer, and see if that holds.  If it does, then piggyback to the next one. If it doesn't, then go back to iterating and see how that goes.  The problem with all of this is exactly what you said, which is that we have to retrain each time, and it's expensive.  But yeah, I think if we can fit stuff on like A40 GPUs, then we can deploy like five of them, and just have like each agent have access.  If to two and we're doing free agents or something, at least we can naively parallelize some of these experiments.

37:08 - Aniket Deshpande
  Yeah, maybe one run pod that each has two A4Ds for each of those, one for immersion baseline, one for backtracking, etc.

37:19 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I'll try and see how much compute I can get to speed that up as much as possible.

37:26 - Han
  Yeah, it'd be nice if we can share a briefing document for this, if we're going to be doing it multiple times in parallel.

37:42 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  What do you mean by that?

37:44 - Han
  The document that we give the agent to let them understand each of the case studies, so we don't write that again.  Yeah, yeah, yeah, yeah.

37:55 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I agree.

37:56 - Han
  We should have thought they're all doing the same thing. Yeah, because we currently have... We many versions of the same case study in different branches, and it's a bit chaotic for the Asian.  But yeah, the branches have gotten out of control.

38:12 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, this is a thing that, to be honest, I don't do very well. But right now, we definitely don't want to synchronize the branches because we're on the hunt.  We're not in farming mode, we're looking for the bison. But then I'm like, yeah, to be honest, most of the weeks we've been on the hunts.

38:38 - Aniket Deshpande
  But also a lot of the code that's in the six branches that Han has and the three that I have, most of it's not what's going to be on the paper anyway.  It's a very small subset of the experiments we're going to use.

38:52 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I know. But there you go. That's So, yeah, I think let's just pivot towards doing these case studies.  I will try and, like, think through a bit more deeply whether there are better case studies to use, but for now, maybe the current setup should be we have tiny stories as our, like, deconfusion example, we have backtracking as a big if true example, emergent misalignment as a big if true example, and the RLHF dataset as another big if true example.  And, yeah, we can just try and do this piggybacking. Maybe this piggybacking is not a good thing and you can't greedy optimize these things.  It's possible. But let's start there and then maybe we can, if it's working badly, go back to just, like, certainly more random parameter search.

40:01 - Aniket Deshpande
  Han, you probably have a lot of the briefing document thing you set up. You probably have a lot already, so whenever you make one for this, can just start running it.  Yeah, sure, I can share it.

40:13 - Han
  I also have a JSON that tries to be a unified source of truth for every architecture that we have, so maybe you could use that, because I found that to be quite helpful for the agents.  Yeah, let me see if I can find it, actually.

40:30 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Maybe you should write a briefing doc for the humans to help them write a briefing doc.

40:36 - Han
  Yeah, so let me just send it in the Slack. So I have this JSON file that should contain the ground truth, the source of truth for every architecture, including the K and the window size and whatever.  So that the agent doesn't just hallucinate those values. you.

41:04 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think that is very helpful.

41:06 - Han
  Yeah, so I'm just going to keep adding new things to this one.

41:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  It lives on the GitHub, or where does it live?

41:18 - Han
  Yeah, I sent a link in the Slack channel. It's on the GitHub, yeah.

41:24 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I guess it's pretty lightweight, so we don't have to worry about it getting very huge. Yeah.

41:30 - Han
  Awesome.

41:35 - Aniket Deshpande
  Something I found funny was you made a branch called Unification, that you're making a new branch for that. Yeah.  Just adding another branch to the chaos called Unification, which I thought was pretty funny.

41:47 - Han
  Yeah, so the intention was for every branch, so all of my branches are going to this one, but my agent started making other branches.

42:04 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Oh my god. Yeah, that is the state of things. Yeah, I think if we had a bit more time, it would make sense to pause at this point and unify architecture and then go forth and conquer.  But yeah, sadly, we're just a week away and that will have to be something for the host deadline.

42:33 - Han
  Yeah.

42:37 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Cool. All right. So let's go with that. I think I will be a little constrained in how much I can be running experiments because I think I should just write the paper now because we do need something.  So yeah, I'll spend a little bit of my time on that. And then once I'm happy with the skeleton, I'll share it and then also...  Thanks for making the time to meet. Thanks for grinding this out. I think on my side, I'm excited about it, but I also appreciate the frustration of constantly having to try new stuff and none of it works.  Thank you for soldiering for it. Thanks.

43:30 - Han
  Cool.

43:31 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  All right, homies. I'll chat to you over Slack. Cheers.