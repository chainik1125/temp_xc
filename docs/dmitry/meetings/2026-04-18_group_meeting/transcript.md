Temporal Crosscoders Project Meeting - April 18
VIEW RECORDING - 81 mins (No highlights): https://fathom.video/share/UncmjeHd_mfqbxPhjm7QA5xnNgKjmYCh

---

0:04 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And just so I have some questions, this is the same cross-coder, that these parameters for this cross-coder are the same as the one you showed the UMAPs for, or is it slightly different?  These are the same.

0:31 - Aniket Deshpande
  They're the same as the GEMMA ones. I think I did something different for the DeepSea Garwin distills. But they're definitely the same as the GEMMA ones.  Yeah, because this is all GEMMA. So basically, I just trained a regular SAE, a layer-wise cross-coder, and then the temporal cross-coder.  The layer-wise cross-coder is on a fire layer window from 10 to 14. And the temporal cross-coder is on a five-position window at layer 12, and I was more interested in seeing between the multi-layer cross-coder and the temporal cross-coder, just to see, because I fixed the parameter counts and the flops for them, the only difference is the access they work on.  Yeah. So I slipped across three different window sizes of the temporal cross-coder, 5, 10, and 20, just to see if window size mattered.  And then there were two ways of matching sparsity that I swept through. One of them was per token K, so basically they all just have the same K value of 100, and another one was total window activation budget.  So because temporal, the temporal cross-coder K is scaled to match the SAEs by T, so that was what Bill did.  And at t equals 5, they end up being the same thing, which is an arbitrary reason they coincide. And so the goal was, are the individual temporal cross-coder features more probing useful?  Which is what the first protocol says, and the second one is more like, is the temporal cross-coder's representation as a whole more useful?  So let me just ask some clarify questions.

2:30 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So let's just report total k, because otherwise I think I'm going to get confused. So the total k for the SAE here is 5?  I think it's 500.

2:48 - Aniket Deshpande
  Yep, it is 500. That is a typo. But yeah. Yeah.

2:53 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And the total k for the temporal cross-coder is also 500. And the same is true for the multi-layer cross-coder?

3:07 - Aniket Deshpande
  Yes, yes.

3:15 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay, so I guess maybe there is... wait, sorry.

3:22 - Aniket Deshpande
  Here k is the probe feature budget. So here k actually is 5. It goes between 1 to 5 and 20.  It's the capacity of the sparse probe. Sorry, sorry, sorry, sorry.

3:37 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Let me just understand. So, like, in a single forward pass, at the encoder, how many features are active for the SAE?

3:52 - Aniket Deshpande
  Here, 5. And for the temporal cross-coder, how many?

4:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  It's five for across all three, in this case. Okay, so we're not scaling K with T in the temporal cross-coded case?  Yeah, yeah, yeah. Okay. Okay, good. Yeah, yeah. Yeah, so this is pretty decisive on just on the layer-wise cross-coder.  I'll ask another question. Do you know, like, what the probe baseline is? Like, how well does the probe perform with this task?

4:46 - Aniket Deshpande
  I think I didn't report that. Yeah, yeah, that's totally fine.

4:50 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  just, the one, let me see if it's in the paper. There would be a mistake. I expect it's basically one, but what task is this?  This is averaged across tasks, right? Mm-hmm, yeah. Okay, can we go to your task breakdown quickly? Thank you.

5:39 - Aniket Deshpande
  The 10-fold cross-coder here wins, I think it was five out of the eight tasks, yeah. It's probably with pretty small margins, and it loses the three most, I guess, relevant ones that it loses are the Europarl, which is like the Language ID one, the Programming Language Distinction one, and then Amazon Reviews.  It holds up against the other two architectures on more topics, like the bios, the topic, and sentiment-based ones, but it's interesting where local information is more important, or local token information is more important with language ID in the programming languages.

6:23 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, it does look like there's not, I mean, there's not a single task that it wins for, right? So I think that's important.  Interesting. Yeah, I was just surprised that the, like, the probing accuracies were quite high. I was, because I thought that the point of, like, the benchmark was to say the SAEs were bad, but I guess the probes probably do slightly better.  But yeah, I don't think it matters a ton. And the layer-wise cross-coder is trained. The layer-wise cross-coder is trained on the same dataset as the temporal cross-coder and the same dataset as the SAE, right?  Yeah. Yeah, good, okay. So I guess even if there's some funkiness about maybe this is not the right task, whatever, they were all trained on the same stuff.  So, yeah, interesting.

7:27 - William Changhao Fei
  Sorry, which layers was the multi-layer cross-coder trained on? 10 to 14.

7:36 - Aniket Deshpande
  Okay. And the other ones are like layer 12.

7:39 - William Changhao Fei
  Yeah.

7:42 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I mean, could be that there's like, you know, layer 12 is not the right layer, but in a sense it's the point of the layer-wise cross-coder that you don't exactly have to be at the right layer.  Yeah, mean, in theory, like the There is a reasonable question, which is, like, what would happen if you disaggregated by, like, the SAE and the temporal cross-coder results by Leia, and, you know, in an ideal world, we would do that experiment, but I don't think I'm, like, hugely optimistic that it will say something very different.  Yeah. Yeah, interesting. Yeah, I guess for me, I'm not so sad that the Leiawise cross-coder beats out the temporal cross-coder.  I am pretty sad that it seems pretty even between the SAE and the temporal cross-coder. Yeah. I'm like, gosh, I mean, if that's the case, like, what is, like, what is the point?  Hello. Hello. Hello. How did these experiments take to run?

9:03 - Aniket Deshpande
  There were a lot of bug-fixing I had to do in the middle, so that took about 8 p.m. to 9 a.m.  I think if everything ran through cleanly the first try, would take maybe 3 quarters of that time, so maybe a midnight to 9 a.m.  thing. So the run time is...

9:24 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I guess the thing I'm wondering is... I did training in the whole run, too.

9:29 - Aniket Deshpande
  So I won't have to do that now.

9:34 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  There are some sort of hacks that we can do for the temporal cross-coder. Like, I think one thing is in some results that I did, it was actually quite helpful for the temporal cross-coder to be at relatively low K.  So, you know, we could sweep over K, we could sweep... We could add like a hierarchy penalty to the temporal cross-coder, which in some of the synthetic stuff that I've done has been quite helpful, but yeah, we may, I mean, if it's like, if it costs us, you know, a quarter of a day to get a result, then we have to be a little bit pessimistic about, you know, how fine a grid we can get, and we should think a bit more about, like, what the, like, what a more principled way of, getting some improvement would be.  Do you have the disaggregation by specific, by specific example? Like, is there a way that we can see for, like, the examples which the SAE got right and the temporal cross-coder got wrong and vice versa?  mystart over. I have

11:00 - Aniket Deshpande
  If the JSON, I'll say, I think I could just make that plot.

11:04 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I would be super curious just to, like, if you, like, could make the overlap plot, like the confusion matrix between the three architectures, but also if there is just, like, the raw text, I think I would be interested to just, like, look through a bunch of examples in each of the, like, entries on the confusion matrix and just see, is there some, like, intuition I can get for when one architecture beats the other, or is it just, like, everything that the temporal cross-coder got right, the layer-wise cross-coder got right, and some stuff that the temporal cross-coder got wrong, the layer-wise cross-coder got right.  And if so, can I understand, like, what the delta is between the two? Right. Okay. I think I'm obviously a little sad at these results, although I think it's really good that we did this measurement.  I think I would be less sad about the results if the errors between the SAE and the temporal crosscode are non-overlapping.  I think if the errors are basically roughly in the same places, then I think the takeaway from this is, okay, we need to have a think about how we're doing our temporal architectures, because the temporal crosscoder doesn't seem to be doing anything that's too interesting.  Right.

12:56 - Aniket Deshpande
  Yeah, yeah.

13:01 - Han
  Yeah, how big are like the variances, like the error bars on this plot, concerns you?

13:14 - Aniket Deshpande
  Oh, I should have added an error bars on that plot.

13:26 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I mean, there are error bars on like, I guess the error bars in one of the other plots that you have, are they across tasks or are they variance within a task?

13:33 - William Changhao Fei
  I think the first one was like aggregated across all tasks. Yeah, this first one was average.

13:43 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But this is like the top thing is, the top part of that bar is like the best performance that the multi-layer cross-coder got on one of the tasks, right?  Yeah, it would be.

13:58 - Aniket Deshpande
  Yeah. Yeah. Yeah. Yeah. Yeah. And...

14:00 - Han
  Did you, like, tune the hyperparameters? Or is there a chance that it's just...?

14:07 - Aniket Deshpande
  It could be. Yeah, I gotta...

14:10 - Han
  chance that the hyperparameters are just misconfigured or maybe it's underfitting, like, severely underfitting or something like that? You mean on the probe side?  The probe side or the crosscoder side?

14:25 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I mean, yeah, we can obviously, like, try to hill climb on the temporal crosscoder hyperparameters, but then it's like, we should also hill climb on the SAE parameters.  Yeah. Yeah.

14:38 - William Changhao Fei
  Something else I was thinking about... I haven't, like, thought about it too much, but it's like... Do you think, like, the temporal crosscoder would, like, have an advantage in, like, earlier layers compared to, like, a sparse atom coder?

14:55 - Aniket Deshpande
  I would think it'd be later layers, just because... I guess that's when the features we helped a temporal crosscutter could pick out would build up in a way.  Yeah, I was thinking about this.

15:09 - William Changhao Fei
  I don't know how important it is, but I was thinking later on, attention already mixes the surrounding information. That's fair too, yeah.

15:17 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But in a sense, I kind of agree with what you're saying, but in a sense, that would be gaming the benchmark a little bit.  Because it's like, I think my concern there is, okay, like let's say we show that the SAE is crap in an earlier layer, and the temporal crosscoder is good.  But really, like if the SAE is better in the later layers, and I think the story that I would have from that is like, look, the useful thing to do is to let attention create representations of the temporal information, and just read that from the residual stream, rather than try to reconstruct the temporal information yourself by constructing an architecture which reads in the temporal information.  So, like whilst like, like, theoretically, it's an interesting question. I think if did that experiment, and then I did the experiment which showed that in a later layer the SAE outperforms, I think I'd be like, huh, that's like an interesting theoretical fact that I would like to understand the theory of, but from a practical perspective, I would be like, the takeaway is still the same as the rough takeaway that I had before, which is let the network construct the temporal representation, so then just read them off at a single token.  I mean, another thing we could do, which is a bit bookie, is we could just like cross-code both across layers and sequence positions, right?

16:53 - Aniket Deshpande
  Yeah, that's interesting.

16:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I guess let's just take a step back here, right? Obviously, we shouldn't over-index ourselves too much to this stuff for SAE Bench, but let's just take a step back and think about the logic of the evidence for a second, right?  So, we have an intuition that temporal structure is helpful in the presence of persistent features. In our synthetic setting, what we show is that increasing the temporal window means that you do better at detecting the persistent features relative to both the SAE and other, I guess, relative to both the SAE and the TFA.  One obvious open question there is, does that remain true for the multilayer cross-coder? So, actually, one thing I would be interested in is if we repeated the synthetic experiments with the...  Okay, we have to do that in a slightly different way. So, let me... Okay, let me note this. So because of our synthetic setting, we can't do a multi-layer cross-coder because we only have a single layer to do, however, there's another series of literature which trains transformers on hidden Markov models like the ones we have and then tries to recover that information about the original generating process from the residual stream.  So for another project, I have some results on that that I can talk about later, but I think I just want to focus on the current logic of our evidence, right?  So our evidence is that in the synthetic setting, the temporal cross-coder should recover these global features at the expense of local features.  When we train these cross-coders and we looked across some of the features, naively, it looked like the temporal cross-coder feature  We were capturing this more like higher-order information across the sentences. Yeah, just one sec, Bill, let me just finish the line of thought.  And now when we've tried to benchmark to what extent is that information useful, on the basis of this one experiment, which we shouldn't over-index to, it seems like it's not that useful.  So I think it would be good to think about each line of evidence in that series of experiments and think, okay, what would tell us the most about whether the thing we ultimately care about, which is how do we build an architecture which allows us to take advantage of temporal correlation, how do we build an architecture that detects that, and to what extent is that detectable at all in language models?  next, when we use Thank you. Yeah, I had a question.

20:02 - William Changhao Fei
  Do we ever benchmark a straight-up SAE, not a stacked SAE, just an SAE on our synthetic data? Yeah, I don't think we did, and I think we should.  Yeah, because that's sort of what we're doing here, but not on synthetic data, right? I agree.

20:18 - Aniket Deshpande
  And should we the opposite of putting a stacked SAE in this comparison? Yeah, we could do.

20:26 - William Changhao Fei
  I guess you could do it, but the SAE here is already, I don't know. Yeah, it's just, like, you could do it, it's just expensive.

20:33 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, you could do it. Maybe it's worth, like, I was going to say it's not worth doing because it's expensive to do, but, like, if your script is, like, you can just run it, and you're fairly confident you won't waste, like, time, like, getting it to work, in which case I don't think it's worth your time.  If you can just, like, run that experiment and, like, get the results without doing anything, then it's worth doing just to help us reinterpret our synthetic data.  go. Alright. But yeah, we should run the SAE, the naive SAE on our synthetic setting, but I think one thing that I would say, yeah, we should run the SAE on that synthetic setting, I would be, my prediction is that the basic story will still be true, and I think what that will tell us, or that what that will suggest to us is the way in which temporal features are present in language models might be a little bit more subtle than the way we've coded for them in our synthetic.  But yeah, think training the SAE on our synthetic setting should take, like, less than an hour, so it seems like on the, you know, insight to effort matrix, it's in the, like, bottom right, sorry, bottom left quadrant, right?  Some insight for very low effort. Yeah, what if it gives us, like, the same, sort of, like, the same result?  Then that's important to know, and then I think we have to think about our synthetic setting again, because I think for me, the narrative of the evidence changes meaningfully if it's like, if I grow K in the SAE, and the SAE improves its recovery of global features, then for me, that means we don't have a good, like, temporal architecture, and we're back to, we're sort of back to the place where we were, I would say, like a month ago, which is like, let's investigate a bunch of different temporal architectures in the synthetic setting, and see if there are any interesting results, because for me, like, the reason why I've like pushed ahead with a temporal cross-coder is one, because it's helpful to have all stages of the pipeline in place to know where to iterate, but two, it's in part because our  Synthetic results made a lot of sense intuitively, and if those results, like, should actually be interpreted as there's nothing the temporal cross-coder is doing in the synthetic setting that the SAE isn't doing, then what's the basis for us to look in large language models other than just like a hope and a prayer?  Yeah, this makes sense.

23:22 - William Changhao Fei
  I can just try to do that right after the meeting. Yeah, I think that would be very helpful.

23:30 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yes.

23:34 - Han
  Haniket, did you use the auto-interp for the multi-layer cross-coder? Like, does it learn different features or... No, but I could just do that pretty easily after I have everything saved.  guess, well, what if the...like, when I was doing the DFA stuff, I just found that the features were just completely different, and maybe the narrative is that all of these methods do different things and we can use them together instead of...  So yeah, Han, I agree with that.

24:08 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But I think we have to be very careful from the perspective of the paper, that's fine. From the perspective of the science, I think we have to be a little bit careful because the counter-argument is like, well, you could always say that for anything, right?  You could always just be like, oh, well, because, like, let me just... Let me just play devil's advocate for a second, right?  Like, I think it's pretty... Like, most of the field believes that auto-interp is garbage. If what we're saying is because the auto-interp, like, and moreover, interpreting auto-interp as a researcher is subject to heavy bias.  So if our claim is, oh, on the basis of, like, looking at the auto-interp for the different categories of features, the vibes were different, then I think, like...  I would feel like, whilst that might get us, like, in the door for a paper, I don't love the science of that.  And so it's totally, but like, like, I'm playing devil's advocate here, right? So I'm being like, obviously a bit provocative.  Like, the serious point is like, to be honest, like, I expect what you're saying to be true. But if that is true, then we have to think really hard about how we can come up with a quantitative measure of that that we're confident in.  And that we're not just, like, producing, like, more stuff for the community to be like, oh, maybe we look into this when we really shouldn't have, right?  So, yeah, I basically expect what you're saying to be true. But then I think the onus for us is to come up with a clear quantitative measure that tells us why that's true, and in what cases one is helpful versus the others.  Which is why I think, like, seeing the confusion matrix for the SAE and the temporal cross. And looking individually through those sentences to see where did one succeed, where did one not, is valuable and is a stepping stone towards this more quantitative way of spelling out in which domains is one better than the other and for what.

26:22 - Aniket Deshpande
  Yeah, and I guess like the argument of if you do want to say that, oh, we could use these like in tandem or like a collaboration within a bunch of different architectures to get the most ideal results, you still have to show that each one has a pitfall somewhere.  Yeah. And a multi-layer cross-coder just kind of blew its capital cross-coder out of the water. Because it's like it outperformed across every task.

26:48 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah. like, okay, that's like, it's not obviously decisive evidence, it's just prima facie evidence that like, okay, maybe this is a little bit subtle.

26:58 - Aniket Deshpande
  And more general.

27:00 - Han
  Sorry, I have one more question. How many layers are in the multi-layer cross-coder?

27:08 - Aniket Deshpande
  It's 5, 10, 11, 12, 13, 14.

27:13 - Han
  And did you pick 5 because you also have t equals 5 for the temporal cross-coder?

27:22 - Aniket Deshpande
  Because SAE and temporal cross-coder are trained at layer 12, so I wanted to just have it around layer 12.

27:29 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I think there's a reasonable question about how does the multi-layer cross-coder scale with its layer width and the temporal cross-coder with its layer width.  if anything, one thing I will say, it sure does look like our temporal cross-coder is under-regularized. So, what you're showing, right, is that as we grow the temporal...  Actually, could you go to that plot again? What is happening to K as we grow the window? Nothing.

28:11 - Aniket Deshpande
  fixed. Okay, good.

28:13 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  K is fixed as we're growing the window, right? Yeah. So one thing that's fair to say here is, well, gosh, if you're growing your window to size 20, but your K is only 5, then on average, you're only getting to decode a quarter of a feature per sequence.  A quarter of a distinct feature per sequence position.

28:36 - Aniket Deshpande
  Yeah, I guess this is more architectural and not email-specific, because a larger T, the shared latent has to compress more tokens into the same K features.  That's right, that's right.

28:46 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think what's an important... Yeah, sorry, Bill, you raised your question.

28:50 - William Changhao Fei
  I think, wait, by K do you mean the K and top K, or the probe K? Okay. Because I think the protocol A, like, also...  Scales K with T, right? Yeah, this is a top K, though.

29:04 - Aniket Deshpande
  Like, top K pre-activation.

29:07 - William Changhao Fei
  Wait, I thought protocol A, like, scaled by, like, total number of features you're taking based on the current, right?

29:15 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  A scaled by total number of features you're taking.

29:18 - Aniket Deshpande
  It's like K with T, right? Protocol A fixes a K. Protocol B takes, like, there's, like, a new K that's K times the feature, the window size.  So in this case, this is a fixed K, but this is the K of the...

29:46 - William Changhao Fei
  Wait, because I thought protocol B kept K fixed at 500, and the other one is, like, 100 times by, like, lane.  Sorry, token lane, sorry, window lane, right?

30:01 - Aniket Deshpande
  Protocol A is the per token K is matched, so they all have the same K. Protocol B is the total window budget is matched, so it'd be K times the window size T.

30:15 - William Changhao Fei
  So there is some K that's changing with the window size, right? Oh yeah, that's in Protocol B.

30:21 - Aniket Deshpande
  There's the total window K, which is, this is kind of like bad naming, but it's the sparsity times the window size, which creates like new budget that you have that changes the window size.

30:39 - William Changhao Fei
  Okay, I think I'm a little confused. No, I thought Protocol B kept it fixed at 500.

30:44 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think maybe, maybe the easiest thing to do is just hold ourselves to the standard that we always report the raw number of K values.  Yeah. And so let's... Let's just go through each data point and say how many k values are there for each data point.  Yeah, yeah.

31:08 - Aniket Deshpande
  Okay.

31:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So, Aniket, would you mind just walking us through that, sorry? Oh, okay. Yeah, so... people can understand, but I just want to make doubly sure.

31:19 - Aniket Deshpande
  Okay, well, this one's pretty straightforward. It'd be 25, 1500. So, don't hang on, hang For protocol B.

31:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Sorry.

31:28 - Aniket Deshpande
  No, it's orange.

31:29 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay, so protocol B is 25, 50, 100. Okay, good. And protocol A is... Is fixed.

31:38 - Aniket Deshpande
  Five across all of them.

31:41 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay, so this is actually really interesting. So what this is saying is... Sorry, let me interrupt this. So, A is 555, right?  Yeah.

31:51 - Aniket Deshpande
  Which is what I'd expect, because you're increasing window size, but fixing... You're keeping sparsity, like sparsity's not scaling with window size.  Because... Yeah. Thank And now you have less accuracy, because you have to compress information further.

32:07 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay, okay, okay. Okay, good, good. So this makes sense. As we increase the temporal window size, what it looks like is happening is K is 5 is too small, because the window size is so large, you have so much information that you're trying to put through the cross-coder, that it's not, like it's not able to be as effective at the task.  However, the gap between them is not huge. Okay, so we have two issues here. Issue number one is the performance degrades as we grow the temporal window size.  Yeah. This is a really important point, because this can tell us one of two things. It can tell us, one, a naive temporal cross-code  It's simply not powerful enough to extract temporal information, or it can tell us we are not using the right hyperparameters for our temporal cross-coder, because what we need is we need knobs that allow us to trade, compute for interpretability, and one of the knobs that you hope for that to be the case in the temporal cross-coder, or two of the knobs that you would hope to be the case, right, is one T and one, sorry, yeah, one T and one K, right, and so what it looks like here is either the temporal cross-coder architecture, as we're proposing it here, is too naive to exploit temporal information, or we do not have the right hyperparameters or an appropriate penalty in order to create good temporal features, and so I think...  think... The takeaway from this graph is we need to think about, and maybe this is something to explore in the synthetic setting, how we can add like an additional penalty or how we might be able to like make tweaks, not like big modifications, but tweaks to the architecture so that we are taking advantage of higher temporal windows.  So in some synthetic experiments that I did, like going to larger window sizes was crucial for recovering like an example.  Yeah, I guess let me just show these results actually.

34:52 - Aniket Deshpande
  Should I stop showing my screen now? You can keep it up for a sec so other people can ponder about it.

34:58 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I'm just gonna find... Oh, and also with the idea of LRT, the shared Phaeton has to compress more positions into the same amount of features.

35:25 - Aniket Deshpande
  I guess there's two reasons that could make this worse, where it's either noise or a dominance, like high variance positions over low variance ones in those K features.  Yeah, yeah, yeah.

35:41 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I mean, and I guess like to some extent, it seems like that might be, that could be a consistent interpretation between why the SAE is achieving roughly comparable performance of a temporal cross-coder.  Mm-hmm. Yeah. Mm-hmm. Mm Good, good. All right, let me sort of like share some of these results which are guiding my intuition a little bit.  By the before I do that, Han, did you ever get the TFA to work?

36:18 - Han
  Yeah, I got it to work, but it was annoying to get it to work. So I posted my findings like just before the meeting started.

36:29 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Would you mind just, sorry, before I show this, would you mind just going through them quickly?

36:33 - Han
  Yeah, so let's go through the four pictures in the Slack. Can you share screen? let's give me a second.  So, yeah, so I ran PFA online. It took some trial and error to get it to run, but I got it to run, and the first thing we noticed is that the features which PFA extracts are either a predictable feature or a novel feature, because each latent is a sum of a predictable part and a novel part.  And in practice, there are no features where the ratio of predictable to novel is like 50-50. It's always like virtually all predictable or virtually all novel, as shown in this plot.

37:44 - William Changhao Fei
  Interesting. me think about that for a second.

37:51 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But that's good, I think. To me, that says the architecture is kind of working well because you... Like it...  I think it's nice... So to have a clean decomposition between features which are predictable and features which are novel.

38:06 - Han
  Yeah, and the novel part is the one that is sparsity-constrained, and the predictable part is dense. So the predictable part is just attention.  Yeah, that's a good point. So this makes it a bit annoying to compare, because naturally these predictable features will have small amounts of magnitude for many, many positions.  So they will be constantly present for the whole window. So most of these are just activating all the time, basically.  Yeah. And now that we do a UMAP of the predictable TFA features, and old TFA features, and temporal cross-coded features, and the stacked SAE features, uh-  We see that they form different clusters.

39:12 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I have to say, this updates me positively on TFA. I guess it makes sense from your earlier plot, but it's nice that the novel and the predictable features are different.  It could just be because that's the case by construction. I guess one thing that's also kind of interesting is it looks like the temporal cross-coder is like halfway between the SAE and those two.

39:45 - Han
  Yeah.

39:46 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Like I have to squint a bit, but like most of the SAE mass is in the top, let's call it the top.  And if I sort of look between the blue blob and the green blob, then I seem to see a lot of...  Yeah, have a clearer plot.

40:04 - Han
  Give me a second.

40:09 - William Changhao Fei
  Yeah, also feels kind of weird that there's like... Oh, never mind. This plot makes it clearer. Oh, this is a nice plot.

40:24 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  This is a nice plot. Kind of looks like Africa. Yeah, interesting. Yeah, it's interesting. I mean, to be honest, maybe I'm reading too much into it, but I feel like I understand you better now when you were like, maybe they're just different and we just should use them differently.  Because I think that's kind That's of the point this plot makes, right? Yeah. I think, so it does look a lot, actually from the perspective of this plot, it seems very clear that the temporal cross-coder is like a bastardization of the predictable features for the TFA and the SAE, which is, to be honest, like exactly what you would expect, right?  Yeah. The thing I would be super curious to do is like actually look at those features in the top right in that predictable corner and see like roughly like, because I think this is kind of interesting because like the narrative that says it's like, well, you can do your TFA, but you, the cost is you have these hyper dense features, but in the temporal cross-coder, you do still have sparse.  There's-local features that tell you some of the stuff that that already told you, but you still have the regular interpretation of the cross-coder.  I have to tell you, this is a really nice plot.

42:13 - Han
  Yeah, and I wanted to do something where I can hover over the features and get an auto-inter, but I haven't...  I did that for the data set.

42:28 - Aniket Deshpande
  I think if you have the data set, you can just put it into an HTML and it'll do it for you.  Yeah. Okay, sure. Yeah.

42:36 - Han
  But I have... Okay, so the next plot I have is I try to look at how long or how long the span of each feature is.  So the span is just... Does it activate across all the window positions or is it highly concentrated in one single position in the window?  Yeah. I looked at how evenly spread out it is in the window, so this isn't a proper analysis because the magnitude is different for each window, so I didn't really think hard about that, but I have one metric where I do a max over the sum of the activation in the window, so if this value is high, it means it's highly concentrated.  If it's value is small, it means it's spread out, so I get something like the temporal cross-coder has more spread out features than TFA and the stacks, but I don't know what to read from this to be honest, I think with the TFA, you would expect the features to be very like, um...  Global already, so they would span multiple positions, and similarly for the temporal cross-coder. But yeah, let me go to the next one.

44:18 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So two things. One, I think we should add the regular SAE to your UMAP. Two, from these plots and taking this plot together with the sparse probing results, it seems to me like one thing that might be happening is we need an additional penalty of some sort to incentivize the temporal cross-coder to read in the temporal information, because it looks like what's happening here is basically the temporal cross-coder is just smearing  ...discovering local features across a bunch of different sequence positions, rather than genuinely discovering non-local features. And if that's true, that would explain why it's not very good.  It's not very good relative to the SAE on the sparse probing benchmarks.

45:27 - William Changhao Fei
  So, yeah, I do wonder whether...

45:32 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I do wonder what would be most interesting to change in the temporal architecture. I think we should have a little bit of a think about that.

45:41 - Han
  Can we not enforce, like, the sparsity of, like, across time, maybe?

45:48 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, so the way, one way to do this was to add a matryoshka penalty. So I like, so I did this for the synthetic setting that I was looking at.  go back. ... ... ... And it roughly, it wasn't super effective, I will show you the results in a minute, but it wasn't super effective, but essentially what it did is it transferred some, it made the features discover more like high-level information in the Markov chain than just a regular temporal cross-coder, and it concentrated them more on single features.  So we could definitely do that, and then we would have a nice story of, in a synthetic setting, we show that this Mataryoshka penalty is important, and indeed when we add it, then that gives us better results.  I would be surprised if it improved the sparse probing results, because the Mataryoshka penalty really is about distributing weight across different sequence positions.  And you know, if we're doing a Mataryoshka penalty for the TXC, we should also do... The other thing that I wonder is, are we sure we don't want to use a different activation function?  So topk is the naive choice, and it's nice because topk is very interpretable when you have control over what the sparsity is.  But we could try Jump Railway, since that's state-of-the-art, it's a pain to implement. Or we could try, like the simpler thing to try would be BatchTopK and see how that changes things.  BatchTopK would be easy to like append to this and might give us interesting results. But I think like once we start doing all that stuff, I think we have to be a little bit careful not to get lost in this like platonic treadmill.  Which is like you just make a plot, you look at it, you have some intuition, you look at the next plot, etc.  Yeah, yeah.

48:02 - Han
  Another thing that popped in my head was, you know how TFA, the predictable component, is kind of like looking at the entire sequence before the current position?  Yeah. Whereas with temporal cross-coder, we just have small windows that we don't know where in the entire sequence these windows are.  And maybe that's not, maybe that's a limitation? Yeah, for sure.

48:33 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah. For sure. And I think, like, I guess from a theoretical perspective, it would be helpful to understand, like, what is that trade-off, right?  Is the trade-off, like, where, like, the temporal cross-coder is more interpretable, but it has a lower window? Or is the trade-off something else?  And I think understanding that would be helpful, because obviously it is useful to read in from the entire... Han, how hard do you think it would be to do the sparse probing for the TFA?

49:19 - Han
  I think it's reasonably doable. think the hard part is just agreeing on a set of things to run, because running the TFA takes a long time, so I need to have a clear picture of what I want to do.  So I think running the sparse probing on the TFA, I think, is important.

49:44 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  If for no other reason than the TFA people didn't do it, then they should have. Yeah, I think, to be honest, it's probably not the, like, thing that's gonna...  Like make or break our paper, I think the thing that's going to make or break our paper is whether the, like we can find this, an understanding of a regime in which the temporal cross-coder outperforms the multi-layer cross-coder.  That's, that's what I think. Like if we were able to make that claim, then other people can benchmark TFA and, you know, be happy.

50:26 - Han
  Um, so yeah, maybe, maybe, maybe, and I was like, you, you mentioned like a few minutes ago, like about what, like doing a multi-layer cross-coder that is like also like multi-position, like, you know, like, uh, having two dimensions instead of one, um, is that, is that like the reason we can't do that apart from, you know, just a compute?  Um, I guess there's, there's no reason we can't do that.

50:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah. Like I, I think there's no reason we can't do that. I'm interested to see what the results would be.  I'm just a bit hesitant to be like, let's run everything and see what happens. But I think it's a reasonable thing to benchmark, I think.

51:13 - Han
  Yeah, because I feel like cross-coding across time and cross-coding across layers are like two ends of the spectrum. Yeah.

51:23 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Well, I guess, so, thinking from the perspective of a paper, I could imagine a paper and, you know, I could imagine an argument which is like, hey, we did this temporal cross-coder, it showed us this advantage, but it underperformed the multi-layer cross-coder.  So let's try and propose this hybrid architecture, and the kumbaya is that, like, the hybrid architecture is, you know, better across, like, most things.  And that is a nice narrative, and that would be a good paper to write. I think... So the more honest, like, current state of the narrative is like, in the synthetic setting, we seem to see some advantage, in the LLM setting, we have seen no advantage anywhere that's not like our vibes of auto-inter, so yes, we could stack these two things together, but like, we don't have a principled reason for thinking that like, that would actually be a reasonable thing to do, but we could, like, again, I'm being a bit facetious, but like, that is, I feel like, at least the caricature of the structure of our evidence, right?  And so it might be, it might be, I think it might be appropriate to take a step back, and just to ask from a more conceptual question, from a more conceptual perspective, like, are we sure that the temporal cross-coder is correctly taking advantage of the temporal information, and if so, how?  And once we understand that, then I think that will give us a better understanding of, like, what, like, what progress could we make with some hybrid architecture?  you. I will say, it is important that we're more labor-limited than compute-limited. If, Aniket, your script is constructed in such a way where we can just propose an architecture and send it off, and then as we do our armchair philosophizing about what the best architecture would be, we can also benefit from experiments running that give us the data for, like, this is a reasonable choice.  Try it, see what happens. maybe that's the way forward, like, let's decide on some metrics for, like, doing a single run so we can do the sparse program.  We will output a confusion matrix, will output the auto-interp, and so on, and then we will go away and think about what is the right thing to do, but in the meantime, we will also have stuff running which is giving us results on our current best gets.  That seems like a good iteration loop to have. Yeah, okay, yeah, sounds good.

54:23 - Aniket Deshpande
  The way I've coded it, there's just a list of architectures, so they all work, and it's pretty easy to plug and play them.  Okay. Yeah, and I think all the bug fixing that I had to do for the run I did last night was pretty useful.  Yeah. I think, like, I had Cloud Code summarized, like, every bug I found, and I think it summed up to, like, 19.  Okay. So, that's why it took, like, an extra three hours, that's what, but yeah, but it was needed, so.  Yeah.

54:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, no, I mean, it's super useful what you did. Okay, so yeah, maybe I think that's... That's the way forward.  Let's submit our current best guess for architectures in the first hour or two of the day that we're working, and then for the rest of our work time, let's think more from first principles, conceptually, how do we take advantage of the temporal information?  Because I think that's the biggest bottleneck for me right now. Conceptually, to me, the biggest problem is our scaling knobs right now don't work.  Our knobs for increasing the compute that we're putting into the temporal cross-coder does not seem to correspond to an improvement in the benchmarks that we currently have.  So either our benchmarks are wrong or the way in which we're constructing that architecture is wrong. So one of those two things has to give.

56:08 - Aniket Deshpande
  What's the immediate idea for a metric for seeing how well temporal information is being taken advantage of, if one exists like that?  Yeah, I don't know. I mean, the sparse probing is one.

56:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Our auto-interp scores are another. This confusion matrix on the sparse probing will also help us a little bit. I think what I would like to do is I would like to construct some more benchmarks, which are like, here is this SAE task.  If we swap in the temporal, which is like broadly temporal, if we swap in the temporal information, like what happens?  But I think, yeah, I will think a little bit about what the right-

57:03 - Han
  So another thing, what I found to be hard to think about is we have so many features these architectures extract, and some of them are more temporal than others, and the only way so far that I know of distinguishing between them is this auto-inter, which itself is not very And I feel like this approach is different from the approach in the TFA paper and in the other paper where they went from the other direction, where they kind of just engineered a data set that has very obvious temporal features, and then they limited their analysis to only those temporal features.  I think that's a reasonable thing to do.

58:06 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I wonder, could we just run the Temporal CrossCoder on their dataset?

58:15 - Han
  Yeah, so that dataset is literally, they took three very different articles and pieced them together into one big chunk of text, and then that was it.

58:30 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I mean, could we just run our architecture on their dataset? Yeah, I think that's feasible.

58:41 - Han
  I need to check if they have the dataset available somewhere. I think that makes sense.

58:47 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I agree with the general scaffold here, which is a good intermediate step between synthetic stuff and in-the-wild behavior is, like, controlling.  So set-ups which, you know, in real models capture the kind of thing that your synthetic setting is meant to be doing, so it could well be that we are missing this, like, intermediate step.

59:14 - Han
  Yeah, like, I think in their papers, one of the features they have are literally, like, are we in, like, a part of the text about maths, or are we in a part of the text about, like, philosophy or something, and I guess those features are very obvious to, like, whoever's reading the papers, that's why they put those features in some, yeah, but...

59:41 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  It sounds like, I think it sounds like we're basically missing this, like, middle step of an engineered setup where we have clear expectations of what the features should be.  So, like, we should think more about what engineered setup we want to use, but maybe a reasonable thing to start with is just running the temporal cross-coder on their synthetic data set.

1:00:22 - Han
  Yeah.

1:00:22 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Like, what I would really like is I would like us to take safety-relevant settings where temporal information might be important, stuff like emergent misalignment, reasoning, and so on.  But I think I'm, like, and I think I will try some of those experiments, but I'm a little more pessimistic about that, given the sparse probing stuff than I am before.  And I think, like, even if those experiments go super well, we do have this issue that our architecture... The is not working for extracting more information as we grow the window size, and I think that is like, no matter like what good benchmarks we have, that is a problem we have to solve because it's pointless having like a somewhat good architecture that you can't scale.

1:01:27 - Aniket Deshpande
  How does the TFA dataset, are they like, are the features labeled or is it like sentence-wise labeling?

1:01:36 - Han
  I think they trained it on like a big dataset and they took like specific samples to show you the difference between the architectures.  And to be honest that paper has many many many different ways of like doing evaluation and some of Some them, I feel like, are very eccentric, or I don't know what they're trying to explain, it just seems very weird, what their evaluations are doing.

1:02:16 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I kind of agree with this. I feel like they didn't do a very honest job benchmarking their architecture, that's how I feel.  But, you know, that's that paper, and we should focus on our paper. But I think, like, I do agree with this general idea that, like, we, like, need this middle step of, like, engineered settings where we're like, okay, here is where the temporal cross-coder should outperform.  Because I think, like, my expectation is still that what, Han, you said is right, that we should find... Some domains in which we understand where each architecture is doing different things, but that's also, like, another way of saying, like, let's have these engineered settings where we expect one to do something different than the other.  Cool. All right. I think, like, this was very helpful. I feel like we're not super clear on what next steps are, so maybe let's just clarify for everyone what we're doing next.  with the, you know, important note that we don't have a super huge amount of time before NeurIPS if we still want to submit, which I think we should.  Yeah. So I guess, Aniket, from your side, I think it would be helpful if... So if you had a script which was like, insert your architecture defined as some class, and this will run a bunch of metrics that we will look at, I think the metrics that seem reasonable, the metrics slash plots that seem reasonable are the sparse probing, the confusion matrix for the sparse probing, the...  We should run the like UMAP clustering. By the way, just as like, like something that we should obviously be tracking is what is the reconstruction loss in each of these cases?

1:04:45 - Aniket Deshpande
  Okay.

1:04:50 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Do you know, like in the results you showed what the reconstruction loss is? It's, if I have it locally, I think I do.

1:05:55 - Aniket Deshpande
  Okay, yeah, I do get that out. I can paste it. Okay, terrific.

1:06:04 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Let me actually write these down.

1:06:58 - Aniket Deshpande
  Okay, so that was with the funnel. The multi-layer cross-coder reconstructs better than the SAE, which is consistent with the probing averages.

1:07:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So basically the reconstruction loss story is the same as the probing result story. Does the temporal cross-coder outperform the SAE?

1:07:26 - Aniket Deshpande
  It's no, not for reconstruction loss, but probe wise it did for five to eight tests.

1:07:40 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay, that's interesting. Okay, so let me just write down like what it might be worth reporting. So I think we should report the sparse probing tasks.  I think the fact that you break down... Break things down into the different tasks is also super helpful. should report the confusion matrix between architectures, we should do the UMAP clustering, we should report the reconstruction loss.  What else did I want to do? Yeah, I think those are the four things. So yeah, if you do it in a way which is like amenable to us going in and just having a class or something which defines an SAE architecture, and we can just run your script on it, which evaluates those four things, that'd be super helpful.  Yeah, sounds good. Han, I think it sounds like the thing that might be helpful... What's is if you try to see if we can run these architectures on the synthetic dataset in the TFA paper.  More generally, I think the question is like, what synthetic datasets can we use? And I think that would be helpful to understand.  So, yeah, if you do that and you're like, okay, like I got the results pretty quickly, then it would be nice to just see what other synthetic tasks can we do in order to try to understand, like, is there, are there different tasks for which the temporal cross-coder is different?  Bill, on your side, I think if you, like, you know, you should run the like SAE tests for our synthetic setting.  Bill, I think the most important thing is the correlation between local and global features. Because that's, like, lot of the basis for thinking that we get… Bill,  more interesting stuff in the temporal setting. I'm just trying to think what else would be helpful. I think it would help if you did a little bit of thinking about what the right hyperparameters are, so I think in particular the thing that I'm curious about is if we, like I wonder if there's an auxiliary loss that we can add that would make the temporal cross-coder take advantage of the temporal information a bit more.  So like the obvious thing to try is like a Mataryoshka penalty, which I think is worth doing, but I think it might be worthwhile just like thinking for a little bit about what the, what a good penalty might be.

1:10:53 - William Changhao Fei
  Yeah, sorry, what do you mean by correlation between local and global features? So if you look in the write-up for

1:11:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  For the thing that we did, there's this plot that Han made, which I think is very nice, which is saying, what is the regression of a probe to the features at a given sequence position versus the underlying latence?  And what you basically see is that as you increase the T in the temporal window, you kind of go, you recover more and more of the global latence.  Okay, okay. I think, so yeah, Han once did that plot, I think it's like Experiment 1c or something, which has the TFA on there as well.  And one reason why I was excited about the temporal cross-code is the TFA didn't seem to do this thing where it did recover the global features.  And so the thinking from there was, well, it seems like these two architectures are doing different things with the temporal information.  So let's just see what that looks like on a big model. So

1:12:03 - William Changhao Fei
  Sure, sure. I'll look more into that. If I have questions, I'll just ask about that specifically.

1:12:09 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah. Cool. Yeah, I think I might try and think of some other settings in which it might be useful to benchmark these two things.  I think I'm still interested in seeing if we can make this work for emergent misalignment stuff. I'm still interested in seeing if we can make this work for auto-research stuff, sorry, for reasoning model stuff.  But I think I have more probability mass in the hypothesis that we may just not know how to train our temporal cross-coder to use temporal information well.  And that might be why the sparse probing gives us the results that it does. I guess the UMAP is kind of telling us that that's not quite true.  And I think the UMAP plot that compares the three different things, the four different things, if you split the other thing, is really nice.  And yeah, if we can understand whether we have those different features, then that would be good.

1:13:32 - Aniket Deshpande
  Yeah, that was something I wanted to bring up, just about the reasoning model thing. I was thinking like, the paper, the base models, know how to reason, thinking models, learn when.  Maybe, because I think that setup, one, they have like a labeled, I think it's sentence level, so it's not feature level, but it's like labeled sentences that we could use for better.  Yeah. Yeah. Yeah. Yeah. Yeah. I guess just more interpretable results, but also they, I think the temporal cross-coder would work better in a reasoning model case because we wouldn't really have to fit our, like in the eval sense, because if for the sparse probing eval, we had to, I guess, kind of like morph, like fit the temporal cross-coder to fit the SAE bench by like introducing these two, those two problems.  protocols I had, but this would be, this wouldn't have that issue. So maybe that could be a more fair approach.

1:14:39 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So look, the experiments are run there is the following, right? Like everywhere where they use a temporal, a SAE, replace it with a temporal cross-coder and see, are you able to do better when you try to reconstruct the reasoning performance from the base model, right?  Yeah. So I think that would be a really, like, I think I would be keen to have you give that a crack if you feel like you have the capacity for it.

1:15:12 - Aniket Deshpande
  Yeah. Yeah, I already have DeepSeq, like, my code can use, already has DeepSeq's, they use the same Arwen distill, so I could, it wouldn't be that hard to let code up.

1:15:22 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay.

1:15:23 - Aniket Deshpande
  Because I, the results I, like the UMAP plots I showed were with DeepSeq already, so. That's right, that's right.

1:15:28 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah. So if could give that a crack, I'd be really, like, interested to see what the results are. Um, again, I, I think a lesson learned is, uh, like, I think one, one piece of clarity that I've received from this is the temp, the, um, layerwise cross-coder is a good, is a reasonable benchmark for the temporal cross-coder.  It's not perfect, but it's reasonable. And I think, for the purposes of our current, and. We should limit ourselves to benchmarking the temporal cross-coder to a regular SAE and to the layer-wise cross-coder.  And I think, although that obviously has the caveats around, oh, well, what about the other temporal architectures? I think I'm reasonably confident now that if we found, like, a context in which the temporal cross-coder outperforms the multi-layer cross-coder, that would be, like, interesting enough.  But also, like, that probably tells us that the temporal cross-coder would do something interesting in that case relative to the TFA anyway.  So, yeah, I think we can just simplify our lives and benchmark to those two things for now, and then we don't have to kill ourselves, like, implementing this complicated TFA.  Yeah. Okay. Does that sound reasonable? Yeah. Yeah, sounds good.

1:17:08 - Aniket Deshpande
  Okay, huge.

1:17:08 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, let's just be mindful of the fact that, you know, the deadlines are coming up. I think I'm a bit more worried than I was before about whether we should submit this to NeurIpsnot.  We have the fullback of submitting to the ICML workshop, but yeah, I think we should check in by next week to see what the state of the results are, and if they're, if we're still super confused, we should probably just, like, write the paper we have, not the paper we want, and then see how we feel about it.  Yeah, makes sense.

1:17:47 - Aniket Deshpande
  Right. Oh, and I forgot to ask, are you, because you said, because you said the last meeting, too, that I guess you seem more confident about ICML and NeurIpsnot.  Is it because you know? Like, the Mekintar Workshop will accept the paper, or, like, Neurop's main track might not, or...  Yeah, yeah.

1:18:07 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I think, like, it's just much easier to get into a workshop than the main track. And it's also, like, from a reasonable person perspective, right?  Like, if our position is like, hey, there's something the community should really know, then, you know, we should submit to a conference.  But if our state is, we're confused about it, we don't know what to make of this, but maybe this is something interesting to the community, then it's a bit more appropriate for a workshop, right?  Yeah, okay. Yeah. Cool. Questions, concerns, comments? None. Huge. Okay, so... Yeah, I think we have somewhat of a sense of what we're doing, but, you know, I think it's fair to say that certainly I don't have an obvious, like, this is how we're going to solve all the problems, so, you know, it's worth being mindful that, you if you have an idea that you're excited about, maybe try that idea, because there's a little bit of conceptual confusion at this stage.  Cool. Cool. All right. If people want to still meet one-on-one, two-on-one, I'm very happy to. If people feel like they have the clarity that they need from this meeting, then, you know, we don't need to have a meeting for the sake of having a meeting.  Yeah. Okay.

1:19:57 - Aniket Deshpande
  Sounds good. See you later.

1:20:01 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So how do people feel? Would people like to meet? Because we have, I'm on the calendar, like I have my, I guess, one-on-one because Andre is at a conference with Aniket and two-on-one with Bill and Han.  I'm happy to meet, but if you guys feel like you are clear enough and you, we discussed most of what we need to discuss here, then that's also totally fine.  Well, I think I'm clear.

1:20:26 - Aniket Deshpande
  Yeah, I think I have everything I need. Yeah, I think I'm good. Cool.

1:20:32 - Han
  Yeah, same. think it's good, yeah.

1:20:35 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think the bitter truths are clear to all. Okay, terrific. Okay, alright folks, then let's hack on this, let's see how far we get, and yeah, let's try not to be like too...  Let's hack on it, let's try and see how we go, and yeah, let's like explore a little bit the space of reasonable options.

1:21:00 - William Changhao Fei
  Okay. Sounds good. All right.

1:21:03 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Thank you, folks. Yep. Bye.