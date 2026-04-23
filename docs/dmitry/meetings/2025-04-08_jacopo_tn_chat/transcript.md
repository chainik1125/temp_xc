Tensor network chats - April 08
VIEW RECORDING - 130 mins (No highlights): https://fathom.video/share/Gz5hmVyTjyfwdBytnKt_BBPzCYyaUjos

---

0:00 - Jacopo Gliozzi
  Good, good. Sorry, I'm late. It is totally okay.

0:04 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I did the terrible thing of asking for a meeting and then not showing up, so I feel like you found the raw end of it.

0:12 - Jacopo Gliozzi
  They're all the same. All the ends.

0:16 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Dude, how's it going?

0:18 - Jacopo Gliozzi
  Not bad, yeah. It's a nice day today, finally. So I biked my extra bike over here so Akash can have it.  I've been promising him this bike for like a year.

0:29 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Mate, how the mighty have fallen. I remember Akash was like to me, oh yeah, I have so many bikes that people are like offering to give me and they got none of them.

0:39 - Jacopo Gliozzi
  I think he had many people offering, but no one falling through. I did.

0:44 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I told him he could take my bike, but he was like, no, I'm good. He even moved that bike to storage.

0:54 - Jacopo Gliozzi
  How about you? What are you up to? Yeah.

0:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  good mate. We just submitted the... war? I don't feel like any sense of celebration, because towards the end, Barry was like, oh, like, what DMRG have you done?  And I was like, who gives a , man? was like, I've done just like open boundary condition, like DMRG.  And he was like, yeah, I mean, we should do like periodic boundary condition DMRG. And I was just like, oh, .  Okay. And I remember there being some issue with it. But it turns out, like, there's no issue. It just converges slower.  Are you doing it in 1D or in 2D? In 1D, in 1D. Okay.

1:39 - Jacopo Gliozzi
  It should still be doable. Like, it's just, yeah, not as efficient.

1:44 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Exactly. Yeah, exactly. Like, we're doing like super small chain sizes. It didn't really matter. But I did redid the like periodic boundary condition DMRG.  And then like, the results look less good. And then I was like, oh, .

1:56 - Jacopo Gliozzi
  Why, why, why, why, like, are these? it's so important to do periodic boundary conditions? Something about the twisted periodic boundary conditions being important or no?

2:06 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I don't think there's any good reason other than the fact that our implementation is PBC. So our clustering scheme is PBC, therefore the appropriate comparison is like the Hubbard model at PBC.  guess you could...

2:23 - Jacopo Gliozzi
  What are you doing DMRG on? The clusters?

2:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, both on the clusters and the total system size.

2:30 - Jacopo Gliozzi
  Okay. Have you ever thought about doing, I mean, I don't want to throw another DMRG wrench into it, but have you ever thought about doing infinite DMRG where you have the unit cell?  I did do that, yeah.

2:41 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So the beef of infinite DMRG is that when we do infinite DMRG, we can't really efficiently do the Aubrey-Andre thing, because when you do Aubrey-Andre you have large systems.
  ACTION ITEM: Remove Jacopo's name from PBC DMRG acknowledgments - WATCH: https://fathom.video/share/Gz5hmVyTjyfwdBytnKt_BBPzCYyaUjos?timestamp=179.9999

2:59 - Jacopo Gliozzi
  maintaining the DMRG, I forgot it was an Aubrey-Andre-like model.

3:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  The periodic DMRG, I need to acknowledge you in the paper because I used your DMRG code.

3:19 - Jacopo Gliozzi
  No, do not let it be traced back to me. Don't worry mate, we'll scrub your name off the gropping paper.  Oh man, I got an email today, I guess you did too, about like something. I thought we'd already lost it.  What is this?

3:37 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I was like, when I submitted the  like HK paper, I was like, oh, thank , you know? Like I can now finally like take a little bit of time to just relax.  And then like, yeah, was a bit on archive. Yeah, yeah, yeah. Oh, congrats.

3:56 - Jacopo Gliozzi
  Oh, that's awesome. I didn't, I didn't catch that. I thought that you just like had one more.

4:02 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I was so sick of it, man. We did so many numerics, man. was like, look at  this Aubrey-Andre modulation, that Aubrey-Andre modulation, in the high U regime, in the low V regime, in this regime.  Jesus Christ, man.

4:20 - Jacopo Gliozzi
  You can say anything you want, but Barry not being thorough, not one of them. He's a thorough guy. That makes the paper stronger, I guess.  Yeah, don't know, mate.

4:29 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  mean, our appendix is basically just, like, 20 pages of plot dump, where it's like, and now we look at this regime, and here is the relative error.  So, like, the main thing we showed, like, relative error, and I'm like, here is the absolute error. You just can't see any difference in the line.  Yeah, funny world. I think, like, man, it was funny. I came back, you know when I came back to Chambonat briefly?  Did you?

5:02 - Jacopo Gliozzi
  When you last came, I remember, yeah, yeah, yeah.

5:06 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I like met with Philip just randomly, and he was like, oh yeah, like we should come to our meetings, and we can do this, and we can do that, and I was like, oh yeah, and there's this open question, and we started like brainstorming HK stuff, and then I like walked out of that meeting, and I was like, what have I done?  I don't want to do any more HK stuff, but I think like, fortunately, Barry's group meeting clashed with his group meeting, so I had an excuse not to go, and now I feel like that this paper is going to come out today, I hope, I can finally be like, okay, you know, I feel like I've said what I need to say about HK.

5:47 - Jacopo Gliozzi
  You will never escape the true clutches of, you know, I feel the same way about dipole conservation, it's like, you know, it's like, all right, I'm done, I'm ready to be done with this, but it's like, my new postdoc advisor is like, meeting with me is like, oh, so, you know,  Like, we're all interested in this dipole conservation. We'd like to do all this stuff.

6:06 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  It really does show you, like, I feel like, again, like, the benefit of the Rajas model, which is, like, you choose, like, one thing that you actually like, and you work on it.  I feel like the downside of, like, mine, I feel like to some extent your model of doing the PhD, where we're just like, oh, let's just, like, do this project to, like, build skills or, like, people out there, because then people are like, oh, yeah, like, it's even, like, people have no reason to work with you, or their reason to work with you is about the...  It's specifically that skill, yeah, yeah, yeah.

6:37 - Jacopo Gliozzi
  Oh, but I'm so much more, you know?

6:42 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Dude, but then, this is when you discover mentoring, mate. Like, I swear to God, like, I've lost...

6:48 - Jacopo Gliozzi
  Your mentoring was to get people to have more compute time on quad, right? That was, that's basically how you described it to me.

6:55 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah, dude, I mean, I'm basically, like, man, mentoring is a great way to, like, have... So a version of clawed code that you only need to intervene on once a day, you know?  Otherwise I have to be sitting there being like, do this, do this. Whereas this way I can just be like, what were the results today?

7:14 - Jacopo Gliozzi
  Do you mean with them once a day? No, no, no.

7:17 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But we like, I usually try and like, it's part time. So like, usually it would be like every day they update on Slack, like this is what I did.  But because part time, I only do it like twice a week. So twice a week, I hope that all four of them have given an update where it's like, I did this today.  And then I'll be like, oh, like maybe try this or like, maybe we can do this instead or whatever.  But yeah, I feel like if you, if you set them up with like codex clawed code, it's quite nice because like, you really can get quite far by just like getting codex to do .  Especially if it's like, you have some clear benchmark that you're trying to beat. And you can do the like error control of like, if this number is but.

8:03 - Jacopo Gliozzi
  I was reminded of our previous journeys in the grokking world, when this morning I'm now like pseudo-mentoring one of Giuseppe's like master's students, and he asked a bunch of questions and we sent him some code to do some sort of like dipole conservating something or something like He asked some questions and then at the end he was like, by the way, like, you know, the code is getting quite unwieldy and like, would it be worthwhile to maybe refactor and like reorganize into some classes or something like that?  And basically I said like, do what you want, but no.

8:50 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Do what you want on a separate branch.

8:52 - Jacopo Gliozzi
  Yeah, do not touch the code.

8:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Oh my God. But dude, mean, does he have access to Claude Code? I have no idea. To be honest, why am I even asking that?  He's Giuseppe's student.

9:10 - Jacopo Gliozzi
  Yeah, probably not.

9:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  No, he seems pretty good at the coding stuff, so maybe he does.

9:17 - Jacopo Gliozzi
  He's quick with it. That indicates something, right? Oh my god, yeah, Jesus. So what is your big idea about entanglement?  Oh, yeah. Yeah, I'm very curious.

9:35 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah, dude. Thank you for asking.

9:38 - Jacopo Gliozzi
  I've been eagerly waiting for like, you know, half a week on the edge of my seat.

9:44 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So in some ways, I feel like I'm going to disappoint you a bit, but in some other ways, I feel like there is, we have done the foundational work for someone who has good intuitions about tensor networks to do even better work.  So, yeah. I'm actually quite excited about this. Let me figure out what I want to share.

10:18 - Jacopo Gliozzi
  This reminds me of a set of lecture notes from this French guy that I was looking at during the postdoc sessions.  He's kind of like a mathematical physicist, and all of his lecture notes are titled, like, whatever the subject is, for and by amateurs.  It's like our tensor network expertise.

10:39 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  If I'm your tensor network expert, oh boy. No, but dude, it's like, you know, there are also like soft skills that come and play.  I mean, I talked to like Daniel Belkin and like, you know, it was on the one hand, I think he had some helpful suggestions, but like just listening to an hour of like, how he has transcended this.  This problem, and how he already has been thinking along these lines, and it's all pointless. You're just like, just shut the  up and tell me what encoder and decoder matrices you would recommend I use.  All right, mate, so with all that said, I do think this is actually a pretty cool project that we're getting some nice results for.  So, yeah, let me just preface this by saying the thing that I would be curious about is just how would you think about designing a tensor network which has design properties that I'll talk about in a minute?  So the question really is, practically, what is a choice of trainable tensor network, of a trainable tensor network ansatz?  other臓s. All Which will satisfy, or sorry, which will encode certain priors about the data distribution that we're interested in.  So that's the concrete task. So like the ideal answer to this is that you, we either have like some MPS diagram, some MPS diagram, or by the way, let me, let me send you this in case you want to write on it as well.  No, not that one, sorry.

12:33 - Jacopo Gliozzi
  No, I don't.

12:40 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So the ideal thing is, and I'll make this explicit in a moment, either some MPS diagrams, so some nonsense like this, and I'll draw you some examples in a minute, or more concretely just a choice of  of what the corresponding tenses look like for this generalized autoencoder. So just keep that in mind. I know that doesn't really make a ton of sense about context, but just keep that in mind, that's the concrete thing that I'm interested in.  So now I'll talk a little bit about the context for the project. So the context for the project is the following.  So right now, a staple of the interpretability literature is this thing called sparse autoencoders. A sparse autoencoder you may have come across before, but it's basically based around the idea of you have some data distribution, you have some vector which is representing some data distribution in a highly compressed way.  And what you want to do is you want to separate out that highly compressed representation into something like its independable components.  rejuvenation. Let's try. try. Let's try. Each of which you hope is interpretable. And so this is like a standard tool that has been around in the traditional ML literature for a very long time, and it was co-opted to provide a scalable tool for interpretability work about two years ago.  It became very popular, then people realized a bunch of problems with it, and the way I would talk about it now is like, we've done the like v1 work of like this paradigm is okay, and now we're doing the v2 work of like, how much better can we get?  And so I'll just talk first about the traditional setting. So the traditional setting is this. You have the residual stream of the transformer, which you can just think about as a vector that encodes the vector of the transformer's working memory.  So the way the transformer works is you have this residual stream vector, you process, you do... So you might have an MLP layer or an attention layer or whatever else you like, but the point is the result of that processing then gets rewritten back to the residual stream and then you carry on.  So it's like a recurrent neural network thing where I'll use A for the activation vector on this thing so you can think about this as like A1, A2, A3.  These are the components of this residual stream memory vector. And so A leaves both a index T, which is the time position, and an index L, which is the layer index, and it's a function of the data input X.  Okay. And so the point is, in an RNN, the activation as a function of T and X is the previous layer.  Yeah. very, at I It's on that activation, plus some transition function which encodes the MLP, or the attention layer, or whatever you like, adding to that residual stream.

16:21 - Jacopo Gliozzi
  Okay.

16:21 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay, just not super important, just necessary background. So what the SAE does is it says, well, the residual stream vector A is a very, very limited amount of memory, with which to do stuff.  And so probably that's a highly compressed representation, which is why it's so difficult to interpret the residual stream neurons.  So let's try and do an SAE to decompose out the different contributions, and hope that the resulting latency- Wait, is this acronym you're saying?

16:56 - Jacopo Gliozzi
  SAE? Sparse Autoencoder.

16:59 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Oh, okay. So the hope is that if you train this sparse autoencoder, then it will be able to separate out the different things that are being encoded within the SAE.  Okay, so how does that work in practice? So in practice, what people do is they train an SAE, which basically is just a encoder and a decoder matrix.  So what happens is you train a latent representation, I'm just going to call it U, and U is some nonlinear activation function acting on an encoding matrix, which may in general depend both on time and the layer of the activation at that corresponding layer, plus some bias.  Yes. And then... And once you have this encoding vector u, you can then decode to recover your best guess, or your best recovered activations, let's call this a tilde, where you just act with a decoder matrix, a function of both L and T on U plus some bias.  So the task of come up with an SAE architecture is really just the task of saying, come up with this w enc, let me give this an enc, and come up with this w deck, that encodes a prior.  About the data distribution that you expect to have, which is going to be helpful for discovering these features. And so, how do you know if your SAE is good or bad?  You typically have this freeway trade-off between reconstruction, sparsity, and what I'm going to call interpretability. And really what you care about is this, so it's not really so much a freeway trade-off as like these two are proxies for this.  But in practice, once you give a metric to interpretability, you do kind of get this trade-off where you can imagine that the more compressed your representation, the better reconstruction loss you get, but then the features just become meaningless.

19:51 - Jacopo Gliozzi
  So it's sparsity, reconstructability, and interpretability. That's right, that's right, that's right.

19:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So, this is the... Traditional SAE setup. And so one point that's important to emphasize is that the input to a transformer is two-dimensional.  The input has a dimension of time and a dimension of space. No, it's the model residual stream. And so you should think about the input to the model, not just as a vector, but that vector over time, because you have this vector, but you feed in different sequence positions, right?  So it's not just the next token. It's not just the previous token that influences the output. It's the sequence of tokens that you're feeding in, which influences the output.  So the input is really a two-dimensional object. Okay.

20:51 - Jacopo Gliozzi
  Okay.

20:52 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  That was a slightly long-winded way of giving some initial setup. And so the current... So thing that people do with SAEs typically is they don't consider this T index.  So what happens is people shuffle all positions together and then they just feed a single position into the SAE and decode at a single position.  So because you do this over a large number of tokens, but you're always feeding in a single sequence position, the intuition to have is that the model is doing the best local reconstruction of the features in the model.  But the premise for what we're doing is like, well, what about global features? What about features that are persistent across different sequence positions?  So the diagram here is if we think about, you know, T equals S1 as being this sequence position, we now also want to know what happens at T equals S2.  at t equals s free, and so on. So essentially we want to understand paths through these activations rather than just single activations.

22:09 - Jacopo Gliozzi
  I understand.

22:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So why would we think that this is an interesting problem? So one reason to think that this is an interesting problem is one generalization of SAEs that performs really well is this thing called a cross-coder.  A cross-coder is a very obvious generalization of an SAE where you read in activations from a number of different layers.  So instead of having just this one activation at one layer, I instead sum over all layers. And this is basically equivalent to feeding in one big activation vector where this is the vector at position a1, this is the vector at position a2, a3, and so on, and you just train this big matrix.

22:58 - Jacopo Gliozzi
  Right. Okay. Yes,

23:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And so that's nice, because if you do this, then you have a shared pool of latents in the encoding across all layers.  So what that allows you to say is, oh, this feature was strong early in the model, it got stronger and then got weaker.  Or this feature became stronger as we added additional layers of processing in the model. So this, I would say, is like a fair statement of the state of the art for single token position essays.  So I think a very natural thing to say is like, why wouldn't the same thing be true if we just cross-coded across time rather than across layers in the model?

23:42 - Jacopo Gliozzi
  Wait, sorry, can you one more time go through that? So for like a concrete example, like the time would be, yeah, like words to generate or something, like phrases to generate.  Which one is time and which one is the layers? Yeah, yeah, it's a good question. Just so I can get my dimensions straight, but yeah.

24:03 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah, so let's, yeah, counting the dimensions actually always good. So let's just say you have this sentence, right?  The cat, sir, whatever.

24:11 - Jacopo Gliozzi
  Yeah.

24:12 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I call, and let's say each of these is a single token. Okay. So I call this sequence position zero, sequence position one, sequence position two, and I'm calling the sequence position as time because it's the time of the rollout of the auto-generated process.

24:28 - Jacopo Gliozzi
  Yeah, okay.

24:29 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And so associated to each of these sequence positions, I have a vector, right? Which is the model's residual stream at a given layer at that sequence position.  Mm-hmm. And so in a cross-coder, what you do is you take a single sequence position, but the model has many layers, right?  So you have like the direct sum. Of many of these at each time, okay.

24:55 - Jacopo Gliozzi
  Exactly. Okay.

24:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So this is like one sequence position. Another thing that surprisingly hasn't done in the literature before, but which we're doing, is you can say, fix the layer and go across sequence position.  And so the point there is, like, we want to discover what are the features that persist over time. Right, okay.

25:22 - Jacopo Gliozzi
  Per layer. But then that's assuming that there's some sort of, like,-wise decomposition of, like, the features or Which may not be necessarily true if there's correlations or something like that, right?  Yeah. And likewise, the other thing, which is, like, fixing a time or something, is not capturing, like, temporally correlated features or something like that.  Exactly.

25:46 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And this is a great point. Like, I feel like in everything that I'm going to say, there are lots of things that we're going to miss.  Like, I think the appropriate thing to remember is the benchmark we're trying to be is the cross-coder, right?

25:58 - Jacopo Gliozzi
  Yeah. Yeah. Yeah. Thank you.

26:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And so the cross-layer cross-coder. And it's not even that like, we need to beat the cross-layer cross-coder across all domains.  We just want to beat it in sum, right? Yeah, yeah. And our thinking is that there are some domains in which the temporal correlations are more important than the cross-layer correlations.  That makes sense.

26:22 - Jacopo Gliozzi
  Actually, I'm surprised that the sparse autoencoder can like, you know, extract significant features of the, of like the transformer's residual stream without accounting for any temporal correlations.  That's...

26:40 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So, so this, this is an important and subtle point. So, um...

26:46 - Jacopo Gliozzi
  Like you, the way you wrote it, you had like a weight matrix or something for each T or something.  Then you're feeding in, like it's being trained on one at a time or something like that, right? That's what you said.

26:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah, that's correct. So, so no... There are some architectures that are a bit experimental, that people have proposed, which I think are not very good, but they're interesting, which try and solve this problem, but the current field consensus is that you train at a single sequence position, but there is a big caveat to that.  You have to remember that when the transformer is doing its forward pass, in the forward pass, i.e. in this function F, we are reading in from all sequence positions, right?  And so I think an important point is that there is implicit temporal information available in the difference between two layers, because that layer will contain an update that incorporates sequence information.  And so there is a good objection to this line of work, which is like... The transformer is the optimal tool that we have for compressing, for coming up with a highly compressed representation of natural language, and we want to take advantage of that compression by just reading off the layers that the transformer has done, rather than like coming up with our own attempted way to read off those temporal correlations.

28:25 - Jacopo Gliozzi
  Okay, I see that, okay. Yeah, so that is an important point in this work. But nevertheless, you may be still missing some temporal correlations, right?

28:38 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Exactly, because the point is that like what we're interested in, right, is like what is the interpretable decomposition of what's happening across layers?  So sure, like the transformer is going to be giving you some update, but if that update is like some highly non-linear function, then it doesn't really matter for the purposes of interpretability, because like we, at that point you might  You as well just train a transformer. Yes, you're right.

29:03 - Jacopo Gliozzi
  From the perspective of interpretability, want to decompose into minimal pieces in some sense, and although the information is all there, you're not really able to interpret it if it's like, okay, I understand.

29:17 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Exactly, and there's a bunch of interesting questions there about like, can you think about attention layers as a generalization of tensor networks?  And it is true that a certain subset of attention layer is provably equivalent to a tensor network, but there are some differences between the two of but that's an aside.  Okay. Cool. Okay, so that's the basic setting, right? And so our basic task is like, can we come up with a w-enc and w-dec appropriately generalized, such that we do better on this freeway trade-off, at least with respect to some categories of text.  with respect to some types of problem okay okay all right so uh what can you do so one really obvious thing to do is like why don't you just do the temporal cross-coder and so like let me so we did that and i'll just show you um to uh you know give you some hope um that what we're doing is not in entirely pointless yeah uh so this is like my favorite graph that we have so far where is it where did it go oh this is because this is the old one yeah sorry  So we have this really nice graph where we compare basically like an SAE trained at a single position to this temporal cross-coder for different values of the window which it's looking back.  Okay.

31:23 - Jacopo Gliozzi
  So the window is like 2 to 10 to 12, sorry. Yeah, yeah, that's right.

31:28 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I think these results are really nice. So what we see is that if we plot out the correlation between the local SAE features and the global features, which are essentially generators of the local features, then if you look at the colors of the dots, what you see is that there's a break between the single SAE, which tracks the local latence, then the higher you make your temporal window,  The better your recovery of the global features is at the cost of the local features. So this is telling you that the temporal cross-coder is basically trading off local feature recovery versus global feature recovery, which is exactly what we'd hoped for.  There's like a nicer way of making this point where...

32:26 - Jacopo Gliozzi
  It's a nice part that even with T equals 2, there seems to be quite a stark difference to the stacked versions of the completely local one.  Yeah, exactly.

32:36 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  It does seem like you saturate if you are getting more and more like temporal positions, which also makes some sense to me.  So the dotted line is like some...

32:44 - Jacopo Gliozzi
  What is the solid line above? I'm not sure.

32:47 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think it's basically like how much correlation you would expect just from doing like a naive probe or something.

32:56 - Jacopo Gliozzi
  Okay. Yeah.

32:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So yeah, like I think the point is that... Like you're just drawing a line, which is basically like how much the local correlations you might expect to be correlated with the glow.  Actually, I don't know, I don't know.

33:13 - Jacopo Gliozzi
  Okay, yeah, some sort of expectation. Yeah, some expectation.

33:19 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, so then we have this nice plot which basically shows that, I guess they didn't include it here. It basically is showing you that you start to do worse on local feature recovery when you start to do better on global feature recovery in the temporal cross-coder, which is great because that's exactly our intuition that the cross-coder decides to encode global features rather than local features, because that makes more sense the more you increase the correlation between token positions.  So I think the exciting thing about this is that even this naive temporal cross-coder is actually outperforming both the naive SAE, which we would expect, but also like a more complicated, like attention-based decomposition that people in the literature have done.  Nice.

34:13 - Jacopo Gliozzi
  That seems like an excellent result. Yeah.

34:16 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  At least in our like toy synthetic setting, right? And so we need to say like, does this hold in language models?  So like one of the mentees is like, the nice thing about this is you can't just do this on a big language model, right?  So like one of the mentees is training, uh, this thing. What is the synthetic toy setting?

34:34 - Jacopo Gliozzi
  Like just, it doesn't matter. I'm just curious.

34:38 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  yeah, yeah. So this is a good question. So, cause in some sense you could wonder whether this is just an artifact of our synthetic thing.  So I pointed blindly.

34:49 - Jacopo Gliozzi
  I'm just curious what are, what is used as a synthetic set setting?

34:53 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I think, uh, to introduce this, I need to introduce like, how people do this in. So in SAEs generally, do the following thing.  You get some random variables, Xi, which are normally distributed with some mean and some thing, and then you also assign a firing probability to every feature, which is like the uniform distribution, say on like 0 to P, where P is your firing probability.  And so what you then do is then you can think about it as like you do some heavy side from FI equals P or FI minus P, let's say.  So this is just a gate which is saying if you draw from this uniform distribution, I guess I'm just going to 0, 1, and you're greater than P, you're on.  If you're less than P, you're off. So there's a gating that happens in these features. And then if you're on, the magnitude that you get is a draw of this random variable Xi.  Okay, okay. So I construct what I'm going to call a... Vector G, whose components are given by this, and then I run an embedding matrix on this vector G to generate a synthetic version of my residual stream, which I'm going to call A, and then I train an autoencoder on this A.  And so this is the single sequence position version of a synthetic setting for evaluating the quality of an SAE.  So what an SAE should do is it should recover a set of vectors XI such that those vectors are one-to-one associated to the ground truth generating.  Oh, sorry. The SAE should have some set of HI whose ground truth- Who- Which are one-to-one associated with the Xi.  So an optimal SAE has a hidden dimension which uncovers the fact that your true generators were these Xi's. Okay.  The reason why people use the synthetic setting is because then you can use the, say, cosine-syn between these Hi and these Xi as a measure of interpretability, right?

37:26 - Jacopo Gliozzi
  How well you can recover the features in this case. Exactly. Exactly.

37:30 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So our setting is a temporal generalization of this. Okay. It's not the only temporal generalization you eat, but it's a temporal generalization.  That makes sense. So what we do is, instead of having these Xi's be random variables, what we instead do is we define a hidden Markov model.  And so this hidden Markov model consists of three things. It's a token vocabulary, a latent space. first The The so Thank  An initialization, and a set of transition matrices for each of the chi. So what is this HMM? The HMM is just some hidden state that evolves under the action of a transition matrix, so like standard Markov model, but the difference here is that in a HMM, your transition matrices depend on the particular token that you happen to emit.  So you have this three-stage process where it's like, one, you have your state vector, your state vector defines a distribution, right?  So you sample from the distribution defined by the state vector, that gives you a token. Based on that token, you update...  The state with a transition matrix, which depends on the token that you got. So in our setting, we have some latent state vector.  The latent state vector defines what was previously the firing probability i. So it's a distribution, let's just call them some distribution.  Well, actually, call them a distribution of the chis. So we, at this point, we sample from the distribution defined by the state vector and we get either a zero or one.  If we get a zero, the feature doesn't fire. If we get a one, the feature fires. And then we update the state vector according to the transition matrix for whether it fired or whether it didn't fire.  Okay. Yeah. So, specifically, the transition matrices that we use here is this. There's thing called a leaky reset process, where the leaky reset takes this form, so basically lambda is like a memory parameter, where in the limit that lambda is equal to zero, you have perfect memory, you always keep the state you started with, in the case where lambda equals one, you have no memory, because this reset matrix always resets you to a given state.

40:24 - Jacopo Gliozzi
  Okay.

40:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So this is nice because this memory parameter allows us to control the correlation between features. I see, I see, I see.

40:34 - Jacopo Gliozzi
  Okay, that makes sense. Thanks for the explanation.

40:37 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, so now we have like a two-way interpretability trade-off where the SAE can either be reconstructing the local features, which we do in the same way as we have here, or it can be reconstructing the global features, which is this latent.  And what we see is the temporal cross-coder reconstructs this latent rather than the local features, which is kind of interesting.

41:00 - Jacopo Gliozzi
  Hope to see. Okay. That makes sense. Very, very cool. Yeah.

41:05 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So where do the tensor networks come in? So now that we've defined this synthetic setting, we have an appropriate benchmark.  What we want to say is like, how can we do the best possible, how can we have the best possible global feature recovery for the lowest possible computational budget?  And so the way in which we would like to do that is we'd like to come up with a tensor network architecture which encodes a useful prior over the data distribution that we're trying to model.  So my belief is that we, although these temporal cross-coder results are already super encouraging, we can do even better by promoting this temporal cross-coder to a tensor network cross-coder.  So, um, because you think that the correlations in time are somehow mediated locally and therefore you

42:00 - Jacopo Gliozzi
  Exactly, exactly.

42:01 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I think the basis for thinking this is that there are a number, so there is a number of works, and I just went to a talk about this last week, where if you model the covariance between token positions as being translation independent, so you just model the covariance as the covariance between two tokens is just a function of xi and xi plus n, and so n, your translation difference, then it's going to be the case that you recover a nice power law scaling of the feature recovery as a function of the size of this covariance matrix that you're allowing.  I see, I see. And so the point here is, I expect the correlations to be predominantly local, but obviously global correlations play a role too, and I think my thinking is that although the tensor Absolutely.  you. you. If may not be the optimal thing for those global correlations, there is some optimization budget that it's applying, and a lot of the optimization budget will get used where it should be used, which is in the local correlations.

43:12 - Jacopo Gliozzi
  Yeah, that makes sense.

43:16 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And so the question now is like, how do we choose, to go back to what I said at the start.

43:22 - Jacopo Gliozzi
  So sorry, so the thing you just said now about the covariance matrix is like, if I were thinking about like, I don't know, like a quantum state, or something like that, that is giving you the probability distribution across different times, or in different sites, or something like that, would mean that it decays also polynomially with distance, or?  Sorry to say that again. Yeah, so what was it that decayed polynomially with, you had some sort of like, power, sorry, power law decay, you were saying?

43:55 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah, let me go back to the paper just so can. I'm just trying to map it back to like my quantum correlations intuition.  Yeah, yeah. Oh, this guy did the paper. Oh, it's with all these people. Okay. Oh, this is a new paper.  I didn't know that. Okay. So it's this paper. Let me actually send it to you as well. It's an Italian person.  Ah. Let me... What do I want? I want to... The way, like, share screen works on Google Maps is the word.  Google Meet.

44:35 - Jacopo Gliozzi
  yeah, I know.

44:39 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So... Oh, yeah. It's a... Sorry. The correlation decays power law as a function of the separation between two tokens.  Okay.

44:48 - Jacopo Gliozzi
  Right. Separation being like temporal separation or Yes. Okay. Yeah. Right, right, right. Okay.

44:55 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So... Yeah. We think that, like, that is a good... Um... A good enough reason to think that a tensor network would be a good way of encoding at least like the zero and first order effects of these correlations, and if we're training these tensor network architectures, there is at least some optimization budget that's going to be assigned to the global features.  Okay, Jesus Christ, let me share a different screen. Sure, yeah, no rush.

45:27 - Jacopo Gliozzi
  Okay, that makes sense. So, I mean, I guess the analog goes, it be like some sort of critical state which also has some power law correlations, but in 1D is pretty representable as an MPS as long as you can tune the bond dimension or something like that.  Exactly, exactly.

45:41 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So one thing you could say is like, oh, but like if the scaling is power law, then an MPS is not good because it's not area law entangled.  And I think my response to that is like twofold. One is like, in practice, you can still do pretty well with a 1D MPS.  But two, even more importantly, remember that the benchmark we're trying to beat is just like a single sequence position, right?  And so it might not be the optimal thing, but it's probably better than that. So let's just try it and see what happens.  So again, the task is, what is the appropriate tensor network generalization of the W-Enc and the W-Dec such that we can use this tensor network ansatz?  So let me give you the way that I've been thinking about it as like a, you know, hopefully as a seed, not a bias to your own thinking.  And so a very crude, but very explicit way that you can do this is to introduce We the network as like a sandwich layer in between a decoder and an encoder.  So the way this works is we say we're going to train a local code for each sequence position in the regular SAE way.  I'm going to drop this layer index because it's going to confuse things. If I can find an eraser, I think just remember that everything is going to be conducted at a single layer in this setting.
  ACTION ITEM: Draft model-diffing plan (cross-layer vs cross-sequence) - WATCH: https://fathom.video/share/Gz5hmVyTjyfwdBytnKt_BBPzCYyaUjos?timestamp=2851.9999  Okay. That's fine, Rumi.

47:39 - Jacopo Gliozzi
  Yeah, I accept. I'm taking some notes.

47:42 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I just had a thought that I want to record for the recorder, which is that it might be interesting to do some model diffing between two different layers in a cross-layer cross-coder versus two different sequence positions at...  The same layer in a cross-sequence position cross-coder, and that might give us some insight into the difference between implicitly encoded temporal correlations versus explicitly encoded temporal correlations.  yeah, for the Fathom note-taker, maybe give me that as a to-do to think about more. Okay, anyway, so we have, we produce local codes at each sequence position in the usual, each sequence position in the usual way.  Again, I'm just going to drop the bias.

48:31 - Jacopo Gliozzi
  Yeah.

48:32 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But then we have a auto-regressive update step where we say, let's define an MPS on the previous sequence, I'm sorry, on all the current local codes to update that sequence position.  So let me give you like a very naive way of doing this. I want to emphasize like, this is the crude way of doing this.  There is a better way of doing this. One thing you can do, I'm just going to draw the MPS associated to M explicitly.  One thing you can do is you can say, I have this, which is a matrix, right? It's the hidden state vector, the local hidden state vector, a bunch of different sequence positions.  Right.

49:21 - Jacopo Gliozzi
  So is that all the sequence, it's all the ones up to T, right? Yes.

49:26 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So that is a design choice. So in the cross-coding literature, you can train something that's called an A-causal cross-coder, which means you read in from sequence positions before and you decode to sequence positions after.  Then you can train an A-causal variant, which is you decode from the same sequence positions that you read in.  So depending on what you're doing, different architectures can make sense. And I think the right way to think about it here is like, this is also a design.  In choice, and we can try both and see which gives us better results. Okay, okay.

50:04 - Jacopo Gliozzi
  So right now you're including some subset of the times, then you can specify, like, if you want a causal or a-causal.  Yes.

50:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, right now think about it as all times are included, we're in the a-causal regime, and then we can drop certain times to go into the a-causal regime, and then we can go into the Go to the causal, yeah.  Okay, so this u, right, is some matrix where I have the encoding at time t here, I have the encoding at time t2 here, I have the encoding at t3 here, whatever, right?

50:43 - Jacopo Gliozzi
  Yeah.

50:44 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So the crudest possible thing I can do is I can design an MPS which does the following. It pairs up sequence-wise each sequence position to say this is t1 and t2.  Thanks to Thanks And then contracts it with another NPS, which goes through another pair. So say this is T1 and T3.  And I can do that for all n, n plus 1 over 2 pairs. Okay. I can obviously choose some subset of those pairs, where the obvious subset is just choose the local ones, right?  So I can think about reading in T1 and T2. Reading in T2 and T3. And here I only just have the n nearest neighbor pairs that I consider.  So this is basically saying there is some two-point correlation that I want to capture, but I don't know where locally that two-point correlation lives.

51:55 - Jacopo Gliozzi
  Can I ask here? So when you're acting... on U with this tensor M, you should get, so U is like a sequence of all of these possible vectors, right?  Yeah, U is a matrix, yeah. Yeah, so then you act on it, and you should get out a single vector, which is like something in the shape of UT.

52:18 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  No, I can, like it's a design choice, right? So I guess the way, maybe the right thing to say is like in the way I've written it here, there would be some projector to time T, which is just read off the like T of component.

52:34 - Jacopo Gliozzi
  Okay, so you're encoding, M acting on U is encoding possibly all of the UT's that you would want, okay?  So that itself should be like basically the same shape as the matrix that you have for U? Yeah, yeah.  Okay, wait, so can you show me like how this, which part is the M and which part is the thing that like the...  Yeah.

52:55 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So this T1 is a, is the... Okay, let me, what I mean by this T1 is this is really UT1, and so this is a vector of dimension H, the hidden dimension, the local, let me call it HL, the local hidden dimension that I'm using for that local code.  Then this is, and so yeah, so M is basically like combining all these pairwise and then training, then I have, so basically the tensors of M, are going to be stuff like M, alpha1, alpha2, T1, T2, and then this is going to be M, alpha2, alpha3.

53:44 - Jacopo Gliozzi
  Okay, okay, I understand now, so then to contract it, you know, by act on U, you put the actual legs there, like you contract those legs with the columns of U or something like that.

53:54 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Exactly, and the important thing is that these are, these are trainable. Right. So I, these matrices M, I'm not like, I mean, there is a question about whether I could use some lessons from like Monte Carlo optimization of the M's because you can make an analogy to the ground state as the loss of the network.  Right. And so you probably can do something clever there, but I don't want to think about it. So I'm just going to treat them as like trainable parameters that I trained via gradient descent.

54:27 - Jacopo Gliozzi
  And right now, like the, this first naive, let's say architecture you're showing me is every pair possible of two timestamps that encodes its correlations equally.  So now you're going to say, okay, there's actually more structured, like, you know, behavior of the distribution. So we should include that in our prior.  And so change the architecture accordingly. That's right.

54:50 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  That's right. That's right. right. Um, so, you know, one way of doing it is just to do these like T1, T2s, uh, and then there's another way of doing it.  Which I always forget, so I'm just going to go to my notes, my notes on this thing, not this one, ju ju ju ju ju ju ju ju ju ju ju, what do I need?

55:26 - Jacopo Gliozzi
  So why not just do like an MPS, so where each dangling leg is like T1, T2, T3 vector, and then there's something connecting 1 and 2, something connecting 2 and 3.  Why have like a doubled input on each node and then connecting it to, so you're like kind of pairing correlations right here.  I don't know, maybe that's what you want to do.

55:45 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah, so I think you can do that. I think my stumbling block is how do I feed, like how do I define those matrices given that I'm doing these local codes?  So because I've done this as a sandwich layer where I first obtained this matrix of local codes, and then I want to take that matrix of local codes and do an NPS step which basically tells us how those local codes talk to each other, I don't know, it's not obvious to me what NPS I would write down that does that, which doesn't pair up these different features.  But I might just I might just be wrong. So let me let me show you like how I should be there and then hopefully you can tell me like what a better way of doing this is.  Okay. Yeah. All right. Bear with me. What do I want? I want to show you, oh,  hell. I forget that there's no way to share VS Code without sharing your entire screen here, so bear with me.  Can you see my screen? Because I can't see it.

57:22 - Jacopo Gliozzi
  I can see it.

57:23 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So here is one way that I thought of implementing the MPS. So I do this local encoder step. E is my, like, W embedding, L is what I called U before.

57:40 - Jacopo Gliozzi
  OK.

57:42 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So E is basically my W enc. And then this produces this local vector. And then I train these MPSs, where I do the MPS.  Over the eyes, and then I decode from the MPS, and so the thing that's confusing to me is, like, you always need some sort, to go from the, like, matrix of local codes to this MPS, you always need some way of, like, mapping that local code information into the MPS.  Yeah. And it's a little confusing to me how I do that in a reasonable way. Here I'm just using the exponential map, but there's, I am a bit unhappy about this, and I don't really understand it that well, and so, yeah, one, if, yeah, if you have an idea for, like, how you take these local codes and convert them into an MPS layer for going cross-sequence position, that would be helpful.-hmm.

58:59 - Jacopo Gliozzi
  So, . Where does the phi come in? So when you have to make this choice of some function to apply to the vectors that you have, these LTs, right?  Yeah.

59:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I think here, this phi is supposed to be some map which takes me from the space of Ls to a space that I can put into this temporal cross-coder.  I think what's happening here is that the phi is being treated as local evidence and then you have this MPS which gets trained separately on the different sequences and then they get combined together to give you an estimate of the decoder.  So basically what What you're saying is like, I train a decoder, I train an MPS separately, I train this local thing, and then I combine them both together so that the evidence that a feature is there at a local point is the product of the five, the five features.

1:00:16 - Jacopo Gliozzi
  I understand. So you need somehow like the MPS to gel well with your decoding procedure. Yes.

1:00:24 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah. And so then there's this other way of doing this. I can't remember what I was doing here. What was I doing here?  What was I doing here? Oh, yeah. Okay. yeah, I guess I, yeah, I think this is a good summary of my confusion.  Like I'm pretty confused about how I take this local evidence and then do this step of piping that local evidence into an MPS.  Where roughly what I want to say is like, the local evidence here is like the physical dimension of the MPS.  But yeah, I'm actually just confused about how to do that.

1:01:13 - Jacopo Gliozzi
  So let me try to be concrete for my own testing of my own understanding. Suppose I have several, like for several different times, I have these vectors L, right?  Yeah. Now, in principle, one could just, yeah, so why not go back to like, let's say the more caveman, as you would say, approach of feeding in the pairs of these vector Ls, like into the, making those the physical legs of the MPS, rather than a transformed version of those?

1:01:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  You can do. You can do. I think in these architectures, what's happening is that you're basically thinking about the evidence in two stages.  One is like a local update, and one is like a global NPS. And then what you're basically doing is you're like, combining those two stages of evidence together.  So I guess like, in what I talked about before, you had like, one sequence of processing that you do.

1:02:23 - Jacopo Gliozzi
  Yeah. Yes.

1:02:25 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  In here, I think the proposal is like, you do it separately. So you do like local evidence, global evidence, then you combine both together as just the product of the two.

1:02:35 - Jacopo Gliozzi
  Yeah. Then in that case, the NPS architecture has to depend. Yeah. Like you said, I guess, yeah, very particularly on the choice of like the shape of the local evidence or something like that.  Right.

1:02:47 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But what I would really like to do is I would really like to think about the NPS, just like the leg going out, the one leg going out is the vector associated with people so, Yeah.  OK, At this really good. Thank

1:03:05 - Jacopo Gliozzi
  That makes sense to me. mean, it's like, that's what the NPS is like trying to encode, right? It's like some probability distribution, which like if you want to actually sample that probability distribution, you just like fix those physical legs, right?  So, okay, okay, good.

1:03:20 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I think this is one way of doing what I said, which I think you might be helpful in seeing like what the issue is.  So what I want to say, right, is like, I have this NPS, which is just the regular NPS, where this dimension here is my, what I call U of T.  So what I need to do is I need to somehow pipe the information from the... u of t into this tensor a of t with alpha 1, alpha 2.  So let's say this is alpha 1, this is alpha 2, this is t, right? So the question is like, what is this tensor?  How do I go from this vector to this tensor? And so it's not obvious to me how I do this.  One thing that you can do is you can leave this as a rank-free object, but then how do you encode the ut?  Well, you can define some w acting on ut, which gives you the t dimension of this thing. But then it's a bit confusing because I'm  Like, yeah, basically I don't know how to go, I don't know how to go from the UTs to the trainable parameter A, like, how do I incorporate the local code information into the tensor A, and it's like, one response is like, I think the tensor network analog is like, you never have this local UT step, you just make the results for like what the local dimension is of this thing, and then that's just your local Hilbert space, right?

1:05:31 - Jacopo Gliozzi
  Yeah, yeah, yeah, I think that's why I was confused about that piece, which is like, that piece is absent in like, let's say normal tensor network things, because it's already set by your local Hilbert space, it's kind of like fixed from the beginning, but...  And to be honest, like, that might be the right thing to do, right?

1:05:47 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  It might be that like, actually what I need to do is I need to just have no encoder step, I just like, my encoder is instead like a trainable MPS, but then that's weird.  Because, like, how do I pipe in the information about the sequence positions, right? Because I have to have luck.

1:06:05 - Jacopo Gliozzi
  Why don't you, I mean, this seems like going to a more complicated setup, but if you could encode the initial information that you're piping in, so these UTs, or maybe UTs paired with the local function phi, if you could encode that as an MPS, so, like, a way to, like, you could, yeah, a way to write that, then what you could act on it with would be, like, a matrix product operator, which transforms it to another MPS, and then that would be the, maybe that's...  Yeah, my confusion is then just, like, how do I... Yeah, yeah, yeah, yeah. I mean, okay, so let's think about, like, what a quantum state is when we're encoding it as an MPS.  So that's a probability distribution that has, like, all these different, like, let's say, parameters, like... S1, S2, S3, S4, and each of those has a local Hilbert space dimension.  Typically, it's fixed across the whole system for simplicity or something, so it's like two on each site or something like that for qubits.  So then sampling this just means fixing some state to take the inner product with, and that state that you're taking the inner product with is some S1, S2, S3, or something like that.  Now, okay, now we have the UTs, so we have a sequence of different vectors, UT, that you're storing as a matrix, right?  Yeah. In what way can we write this as a probability distribution, can we?

1:07:48 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Well, so the way to think about it is like the Hamiltonian is like the set of weights in the transformer, and the wave function is like the

1:08:02 - Jacopo Gliozzi
  Right, if the wave function is like the residual stream, right, which is then getting, you know, to, sorry, forgot the notation, but UT is not, UT is like some encoding of the residual stream, right?

1:08:15 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  That's right, and I think the challenge is that, like, what we're trying to do is, like, we don't want to work with the residual stream, we want to work with an encoding that is, like, hopefully more interpretable, right?

1:08:26 - Jacopo Gliozzi
  Yeah, well, okay, I guess it really depends on the shape of the encoding, but suppose that you were working with the residual stream directly, then that would be some probability distribution, which you could, like, feed into a matrix product operator, or represent in a compressed way as a matrix product state, the residual stream itself.  So, now, if that residual stream is getting encoded, as a sparser, or a smaller, you know.

1:08:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  My confusion is this, like, I think it's the same for wherever you're. Thinking about the residual stream or the UTs, right?  It's like, so I have, so I guess we can think about it this way, right? I have like a probability distribution, let's just call it X, which is basically like a probability distribution over some vector, right?  That vector, you know, A1, A2, whatever. And so the question is, like, if I go over all sequence positions, then going over all sequence positions is a distribution of that vector, and the question is, like, what is a good ANZATS for encoding that distribution?  And I think what I don't know is, like, how do I convert the temporal analog of this, i.e., this at a bunch of different sequence positions, directly into an MPS?  I think that that's, I mean, why don't you...

1:10:00 - Jacopo Gliozzi
  Take the tensor product of all of these vectors, right? The tensor product or the direct sum? Let's say tensor product, right?  So pretend that each of these vectors is the on-site wave function of your system, right? Actually, it can be something that's not a tensor product wave function.  That's the broader point, I guess.

1:10:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  It might approximately be if they're, like, independent at each sequence position. Right, yeah.

1:10:33 - Jacopo Gliozzi
  But, like, suppose you construct an object which has the shape of the tensor product of all of these wave functions, right?  This is like a quantum state and that you can, I mean, a class of, like, you know, quantum state squared or something like that, which is, like, the probability distributions.  And this thing you can compress as an NPS into those, like, different points, right?

1:10:55 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I guess my question would be, like, what is the A matrix? Like, what is the AT? alright. Okay. Yeah, you have to, I mean, how do I, I guess the better question is like, how do I train to obtain the AT alpha one, alpha two matrix in that case?

1:11:12 - Jacopo Gliozzi
  So this one I think would be not something to, I think this is like some, let me see, let me see if I can write on the thing one second.  So yeah, I have to reload the whiteboard. Okay, here we go.

1:11:31 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  One second, I have to let someone into the office. Oh yeah, go, go, go. No, no, no worries.

1:15:12 - Jacopo Gliozzi
  All right, so this is my, let's say, baseline to set my notation, the product state. So it's NPS with bond dimension 1, so you can kind of, you know, separate them.  But here, I have my probability distribution psi, which is described by all these numbers, and in particular, like, some special tensor product of these numbers.  Yep. But then, like, it comes from separate distributions on each site.

1:15:43 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But I guess I'm a little confused, like, this is the, I mean, this is a special case, right?

1:15:50 - Jacopo Gliozzi
  This is a special case, yeah, this is a very special case, exactly. So this is like the, like, like, you know, naive setup, which is like, I have some problems.  And what does it mean? This is a probability distribution over these variables s, which could be the spin. I'm calling the spin on each site, right?  So on each site it could be 1 or 0, right? That's why it's a two-state vector or something like that.  But for this one particular form of the probability distribution where it completely factorizes, then this is the NPS for it and this is how I would extract it, but it doesn't have to factorize, right?  It could be some other probability distribution. But the important thing is that it's a probability distribution over s. I think that that's the key thing here, which is I have some distribution and I can sample it in this way.  So these two steps here don't depend on the fact that it's a product state.

1:16:48 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yes. Yeah. I see what you're saying. You're saying I have a probability distribution as a function of the size 8.  The probability distribution defined by the NPS is just a function of that size 8 vector rather than the size 6 vector coming from the 2D vectors at each point.

1:17:15 - Jacopo Gliozzi
  Yeah, yeah, right, right, right, right. I guess, yeah, the example of three sites or three times is like a little bit like small, so like 8 versus 6 is not that different.  But like, eventually, like, the thing is, like, I have a probability distribution that in the quantum case is like way, you know, yeah, I have two to the time steps different options for like, my probability distribution, like in some senses is, you know, my, this is the distribution.  And then these are all the different bit strings. So like, these are, you know, 010 all the way up to like 111 or something like that.  And there's two to the number of time steps up here, right? Yeah. So what is the, so, okay, so, and then.  This will look something like that, and my goal is to sample it, which means like, you know, given some access point I need to know, kind of draw this point with this probability, right?

1:18:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I see, so you're saying like the probability distribution that you're considering is the probability distribution over all possible bit strings.  That's right. This is an efficient way of writing down the probability distribution over all possible bit strings, i.e. there is some map from your tensor network to each bit string, which gives you approximately the correct probabilities.

1:18:41 - Jacopo Gliozzi
  Right, and like my space is like for each element in my space, I can decompose it as like something like this, where I have some option to put like in the first site, some option to put in the second site, and some option to put in the third site or something like that.  Yeah. is like one, site two, site three. So this is like some, you know, element of my- space that the probability distribution is defined on.  So now let's try to make the analogy to your data coming from the residual stream and the encoded vectors.  So I guess the point is when you have that matrix of encoded vectors, I was naively just likening them to this would be like you at time one, this would be like you at time two, be like you at time three.

1:19:28 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  That's kind of the analogy I want to make. Yeah.

1:19:32 - Jacopo Gliozzi
  But then like in this kind of direct analogy, I mean, I feel like the only probability distribution that I'm creating is like a factor distribution.  That's right. But of course, like the point you're saying is that the U1, U2, U3 like are already encoding some sort of correlations.  Well, okay, so let me just be clear.

1:19:51 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  The U1, U2 and U3 is a design choice where I wanted to have some notion of what the local codes were.  And I wanted to use the tensor network on the local codes as opposed to just using it directly. I think you're proposing that we can do it directly, and I think that's the more elegant way of doing it.  was just harder for me to think about. So my point is, there's no reason why you have to think about these as like U1s, U2s, U3s.  You can just think about this as like A1, A2, A3, if you want.

1:20:28 - Jacopo Gliozzi
  Those are the activations of the residual stream. That's right, that's right.

1:20:31 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  At different time steps.

1:20:33 - Jacopo Gliozzi
  Yeah, but, okay. But now like, I guess, honestly like, you can always represent them as a product state. Then, like, just like as a formal thing, which is just like, these vectors together, you create something which is a product state.  But there may be correlations between the vectors. So like, in the quantum case, right, like, let's suppose I have this vector.  Here, which is like, you know, the spin-up vector, tensor, the spin-up vector, tensor, the spin-up vector, and so on and so on, right?  So like an all-up state. Yeah. So here, actually, like every, the state at each site, like site 1, 2, 3, 4, they're all very correlated with each other, right?  That's right, that's right. But the NPS is, like, completely uncorrelated because there's no entanglement between sites.

1:21:32 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Well, the NPS should be correlated insofar as you have some non-zero bond dimension, right? So this is, this has one, bond dimension 1, right?

1:21:41 - Jacopo Gliozzi
  That's because it's a product state. Right. So in this sense, like, there's a correlation between, like, the, the value of, like, the local, like, my distribution is correlated, but it's not entangled in this way.  So that's, that's, that's, I think, some pitfall, which is, like, entanglement would be, like. Capturing something about two sites are somehow non-independent, or the state on two sites is not independent.

1:22:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Right, which is I think what we want to capture, because my point is the standard SAE setting is basically this UNSATS, because you're assuming there's no correlation between token positions to employ, and now we want to generalize that to say, here is an architecture which is still quite efficient to train, but can tell you something about these correlations between sequence positions, right?

1:22:34 - Jacopo Gliozzi
  Right, yeah, so I guess like the quantum statement here would be like, if I make a measurement on this site, it doesn't tell me anything about this site.  Yes, exactly. But in your, but that's, you know, wrong, right? Like this should be, this should tell me something about the other site, because they're related by some correlation.

1:22:53 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Exactly, exactly, because there's some feature which lives at one sequence position which persists into the third sequence position. Yeah.

1:23:00 - Jacopo Gliozzi
  So then, I mean, maybe, yeah, maybe you should just allow, like, a free set of parameters, which is, like, your NPS, I don't know, like, some NPS with, like, some bond dimension that you're choosing, like you were saying, that's just, like, a free parameter that you're tuning, and the values of these matrices are also free parameters.

1:23:16 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But then how do I, how do I get that NPS to reconstruct the activations? Like, how do I pipe in the information about the distribution over activation vectors?

1:23:27 - Jacopo Gliozzi
  Yeah, mean, it's, like, like, you can create, like, given all of the coefficients of the wave function, so all two to the, like, number of time steps, you can create an NPS at some bond dimension by, like, going site by site and doing SVDs, right?  That's, like, that's, 10pi has that algorithm already there. But that would be, like, you have to start from some distribution, which is, like, your probability distribution that depends on, like, you know, the, some options at, at time one.  you. Then some options at time two. So like in that sense, you need to kind of start with the exact probability distribution and then you can like compress it to an NPS.  But that's not really what you like that. I don't know if that would really help you. Right. Like, I guess you'd want to kind of.

1:24:17 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Well, how like, I guess, I guess what would I be like, what's the approximate dimensionality of these options? Right.  I mean, if I can feed in like a very large number of options, then that's good. Right. I mean, I basically that that would if I basically like if this P op one op two is basically my training pipeline where I say like gradient update from each from each sequence position, then that's what I want.

1:24:48 - Jacopo Gliozzi
  Yeah, I guess. I mean, like, so one thing is like that these things all have to correspond to the time steps.  Right. That's the. Well, there's two.

1:24:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  No, there's two. There's also another act. Yeah, there's like a window of time steps that you feed through, and then there is the number of sequences that you have, right?

1:25:09 - Jacopo Gliozzi
  I see.

1:25:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I mean, this must be something that we do in the many-body setting, right? It's like if you have a many-body system, you're taking many samples of the wave function where you can do it.  I it's basically the problem of like, let's say you have a 1D chain, you want to take local measurements of that 1D chain, you want to do many copies of those local measurements in order to come up with the best MPS reconstruction, right?  We're basically doing the same thing here, right? Where like our chain is the number of, like the window size, and we want to take, and our number of samples is the number of data points that we want to run through, and we want to come up with the best MPS reconstruction of that.

1:25:52 - Jacopo Gliozzi
  Okay. That's, yeah, that's an interesting problem. I don't know. I mean, it surely must be like solved. Somehow in the quantum literature that like given some measurements, what's the best possible NPS reconstruction?

1:26:06 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I remember there was some great MLP, which tried to do this for phase transitions.

1:26:16 - Jacopo Gliozzi
  Yes, I mean, we didn't solve this problem, probably, but it's difficult actually, like given the measurements to, yeah, I don't know.  I mean, maybe you're right, like maybe this is too, like, on a broad, to do it on such a broad scale, it's too difficult to compare the architectures, but it's better to have like the goal that you're saying, which is like use some NPS architecture to like encode a piece want to do is I want to use the NPS as like a pipe for information to go between sequence positions with the idea that like, that the NPS in like allows us to encode the like assumption that that I don't

1:27:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  The is largely local. It's not fully local, but it's largely local.

1:27:04 - Jacopo Gliozzi
  Yeah, I mean, that really does make me think of, like, some, you know, case where you have some information which is stored as an NPS.  Don't, you know, don't ask me how right now, but, and then the, you know, the pipe, the local pipe is some operator like this that maps it to another NPS or another similar form, but in a local way.

1:27:25 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I guess, so I guess what you're, I think what you're proposing here is, like, the fully tensorized, but, like, generalization of the SAE problem where, like, you start off with an NPS representation of the initial state, your encoder is, like, this NPO, like, evolution, and then you decode to some NPS, and then you further decode to, like, the local activations.

1:27:52 - Jacopo Gliozzi
  Yeah, yeah, I mean, the point, problem here is, like, it doesn't really do any sort of compression, like, in a, I mean, the actual, actual.  If of the NPO and the MPS will just give you another MPS, then you can do a compression yourself, like with some SVDE or something like that.

1:28:07 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah. Basically, I think the problem in, not the problem, but I think the obstacle in what you're saying is we need to go from the sequence of activations to some MPS, which itself would be V1 of this Tensor Network SAE, because, like, getting that representation in a trainable way is, in a sense, what I was trying to do with this, like, loading up of, like...

1:28:38 - Jacopo Gliozzi
  Right, if you solve that problem, then you've already, yeah, yeah, yeah. So, okay, so suppose you do this thing that you were saying, you know, like, this pair every UT with every other UT, right?  Yeah, yeah. What's the problem with that?

1:28:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think, as I'm saying this, I thought I had a solution to this, but I... I actually think I don't.  I think the problem with that is I'm not sure how I go, like, what is the actual A, like, what is, how would I write down this?  Yeah. What would the actual elements of that be? Because it's not, I'm just like, I somehow have to, like, construct the map from the residual stream to the M, which knows a little bit about the residual stream, but which is not, like, explicitly those coefficients, because then, like, what are my other coefficients in this thing?  But, sorry, but can't you just leave M to be unconstrained and then optimize somehow? Well, I still have to feed the information about what the residual stream is there, right, which is why I'm saying, like, I need some map to, like, go.  Maybe that's the thing. You can also just leave the map as a free parameter. So maybe the thing to say is, like, well, let me do it here.  Maybe the thing to say is, like, I start off. With some U, which has some index T and some index H for the hidden dimension of the U, I apply some like W, H, I, which maps me to the space of this, which I can track with this like AI at time T.  And which has some alpha one, alpha two. So I guess the idea is like, I have this vector, I contract this vector with some W matrix, which then leads into this A, which has a leg for...  Another tensor. So at each point I'm like doing something like this.

1:31:08 - Jacopo Gliozzi
  And I guess I want like a free leg here.

1:31:13 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Maybe. I don't know. I'm a bit confused by this, but this is, I think it's something like, cause then I can like, if I have this matrix W, right, then I don't have to, I don't have to have any trainable, I don't have to have any, I don't have to explicitly set the coefficients of this AI.  I'm just like, I have this W, which is, reading in the information from you in a way, which is learnable.  Uh, and it's feeding that information into the A in a way, which is learnable.

1:31:43 - Jacopo Gliozzi
  where do you get the W's from? Sorry. The W is just another trainable parameter.

1:31:48 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Oh, okay.

1:31:49 - Jacopo Gliozzi
  Yeah.

1:31:50 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Um, so, and I, in fact, I wonder if I can just do this directly and have some new HT, uh, and so.  Yeah, I guess I could do this, right? I could just have this, and then the tensor network is like I have these A's, which I can track the alpha 1's and alpha 2's over, and I guess like in this case, instead of having like a free parameter, instead of leaving the local Hilbert space free, I'm basically saying, the local Hilbert space is contracted with the U, which is passing my information, which is like in the free index.

1:32:42 - Jacopo Gliozzi
  Yeah, okay, actually that makes sense, because I mean what you're trying to do is like extract features, right? So like you're just like trying to group together different U's with some set of coefficients which are encoded in your A here, and then that will be the way you're encoding like your D.  Okay. So maybe that works.

1:33:03 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I guess this should have a free index here as well.

1:33:09 - Jacopo Gliozzi
  Yeah, sorry. So like, I guess you can, yeah, you can do that. And then you create some broader tensor, which is this contraction of U with A, and then that's the thing that then you somehow have to decode.

1:33:20 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Well, yeah, I think the decode is just the opposite of this, where I have this like A, and I get like a decoding matrix, which maps me to some like 1D thing.  I mean, yeah, I mean, I think I can always just take this and say, actually, that's a good point.  I guess if I'm contracting over H here, and I'm contracting over alpha one and alpha two, this does just end up being like a scalar.

1:33:47 - Jacopo Gliozzi
  But that's kind of like, wait, I guess, if you have all these like U's coming in, right, like that you have a lot of information, like U1, U2, or something like that, and you're trying to compress them, you know, and you should  We should somehow, like, contract them with each other in a smart way with some coefficients given by operators, you know, A1 or A2 or something like that.  Yeah, yeah, yeah. And then this will kind of hopefully eventually reduce the size of your data that you're inputting, which is the U1, U2, U3, U4, right?  And then this thing will somehow encode some correlation features. You don't want to, I guess, just do some invertible map, because then in that case, it's just the same information as the matrix of U's, right?  Yeah. So I think what you proposed is quite reasonable, and it is, like, know, tensor networking in the sense that it retains, like, a local bond dimension that controls how much you're pressing.

1:34:58 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I guess, I guess, what's not clear. I to me is like what is the dimension of the like latent space here because like if I do this contraction the only three indices will be the first and the last alpha and so I'm not really sure how to interpret that right because really what I want is like I want a vector which is like the features associated like I want some sort of notion of like these are the global feet but I guess in a sense that's yeah I want some notion what about like the multi-scale entanglement renormalization thing like the the one that looks like a tree that is a tensor network that like I guess looks something like this and then these things are connected there's some tensor here there's some tensor here there's some tensor here then like these things are connected there's some tensor here I guess yeah my question is again like what is the end  Yeah, there's still a, hmm, let me think.

1:36:10 - Jacopo Gliozzi
  Yeah, depends. guess it depends on how many layers you run, right? Then you can leave some dangling legs out here.

1:36:18 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, but what is the input? Sorry, yeah, yeah, yeah.

1:36:25 - Jacopo Gliozzi
  Wouldn't the input be, like, your U1 here, U2 here, right? Yeah. I'm thinking of, like, contracted version, right? So then now, with these vectors, I'm performing some sort of tensor contractions with these tensors, which I don't know what they're going to be, but...  It can be trainable.

1:36:47 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, you can adjust them.

1:36:48 - Jacopo Gliozzi
  And then, depending on how many layers you do of this, you can, like, end up with, you know, just two output vectors or something like that instead of...  Okay.

1:36:58 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay, that sounds...

1:37:02 - Jacopo Gliozzi
  This is called this Mera. But I think usually people, the way they actually do it is they have an MPS with dangling legs, which are the ones that you have to contract against.  This is supposed to represent a state. So right now I'm not such a very big vector. And then here they have all these tensor contraction things.

1:37:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  In theory, Mera is good if you have power law scaling, which is what I'm Right.

1:37:32 - Jacopo Gliozzi
  Yeah, that's right. That's right. That is what it's designed for. But I mean, the fact they have power law scaling means that like a scale invariant kind of onsots is supposed to like efficiently capture the structure of the correlations because you have correlations at this scale and also at this scale and also like this scale.  So in some sense, like this would generate the kind of compression that you're like the change in space that you're looking for without.  Like, just contracting it to a single number, maybe, and you have a tunable parameter in addition to all these tunable parameters of the tensors, which is, like, the number of layers that you're applying.  So you should apply, like, kind of few layers, otherwise it will become very, I mean, you can compress a lot, but then you have a lot of free parameters of all those tensors, right?

1:38:19 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I guess the beef I have with this is, like, this seems like a more complicated way of implementing the tensor network idea, and it would be good if I understood just, like, a choice.

1:38:33 - Jacopo Gliozzi
  network idea itself, yeah. To begin with.

1:38:38 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But, I don't know, maybe we can, like, solve it from the other direction.

1:38:44 - Jacopo Gliozzi
  Yeah, no, I mean, this, like, I was thinking of because, like, you wanted some way to, you know, reduce this sort of information that you start with and code some correlations, because it's, I mean, it's not that, like, you just have  Like in the quantum sense, this is not, you shouldn't write this as like a quantum state or something like that, because then this would just be like a product state and, you know, you wouldn't gain anything by just like doing some sort of tensor network representation, like it's already in a product state representation, but instead you want to reduce the information somehow further, which I'm not sure how, but yeah.

1:39:25 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I think I've, yeah, so I think the, like the objects that I want, right, is like, I want some set of like hopefully interpretable latents that I'm extracting from this, ideally like with some notion of like, these are the local latents and these are the global latents, and then I want to decode to the initial object that I started with, like those are the two like properties we need, right?  And I guess like, and the thing that's confusing me about this is, although this seems like the right thing to do, it doesn't seem like I have the right number.  Like in the quantum sense, this is not, you shouldn't write this as like a quantum state or something like that, because then this would just be like a product state and, you know, you wouldn't gain anything by just like doing some sort of tensor network representation, like it's already in a product state representation, but instead you want to reduce the information somehow further, which I'm not sure how, but yeah.

1:40:25 - Jacopo Gliozzi
  Yeah, I think I've, yeah, so I think the, like the objects that I want, right, is like, I want some set of like hopefully interpretable latents that I'm extracting from this, ideally like with some notion of like, these are the local latents and these are the global latents, and then I want to decode to the initial object that I started with, like those are the two like properties we need, right?

1:40:49 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And I guess like, and the thing that's confusing me about this is, although this seems like the right thing to do, it doesn't seem like I have the right number.  Three legs to have a vector in that intermediate step.

1:41:06 - Jacopo Gliozzi
  And so I don't know what my analog of like latencies in this case.

1:41:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  This case, I think I on the mirror case, I think I understand a little bit better how you get the intermediate dimension.  But I like don't have any like I just have like a bias that this seems like a more complicated.  Yeah, well, what if you just do like one, let's say we're doing like a tensor, we're returning to more tensor network-esque structures.

1:41:34 - Jacopo Gliozzi
  So you have like your U1, your U2, your U3, and so on.

1:41:41 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And then all you do is just have a tensor here. So I guess this would be the latent space that looks like that.  Yeah, okay, good. So I guess let me just think, okay, good, good, good. So this is the kind of thing that I want.  Now my question is... Is what are these, how do I find these matrices here? Well, I would say, yeah, okay, so that shouldn't those be free parameters or no?  Yes, but then how am I incorporating the information from these U's? So is the idea here that like, I start off with, because this to me seems like this.  Yeah, yeah, yeah, I think so, right? So then the question is the same, right? It's like, what are the free parameters in this case?  I guess here you have this additional leg here, right? Yeah, right, it's taking two states, right, and giving you one somehow.

1:42:41 - Jacopo Gliozzi
  Yes, I guess, let me just try and write down the explicit tensor for this, right? So here, I think you, let me call this delta T since it's indexed by the gap between these two things.  And so I have some. And I guess

1:43:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Alpha 1 here, some Alpha 2 here, and then I have an additional index, which is like, I don't know, D?  The Alpha index, yeah. D1, and so I have to contract this with like U Alpha 1 and U Alpha 2, T2 and T1, and then the thing that I get out is this new local representation, D1.  Right. So this is good. The thing that is a little upsetting is that I know... dimension, by the way, I guess.  What's that? I wouldn't know what would be the bond dimension here. Right, right. But even so, I mean, this is not a terrible thing to do.  D1 is sort of like some analog where if you tune how much, like how big, you know, like the size of the index D1, bad, Thank

1:44:00 - Jacopo Gliozzi
  And you can encode better or worse, right? Yeah, I guess what I would want is like, I don't want like a separate vector space at each of these like gaps, what I want is like a single vector space for the whole thing, but maybe that's just conceptually not right.  So yeah, so a single vector space for the whole thing, I guess what you can do is like, then sum up the vectors you get at the outside.

1:44:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But I guess, I guess the beef here is like, what I, but I guess maybe this is okay.

1:44:33 - Jacopo Gliozzi
  I guess what I was going to say, like my beef here is like, what I would like to say is like, there is some local, like these are the local codes.  Yeah. And this new object incorporates both the local codes and the global codes, but instead it's like, this is encoding really just these two, like information from these two up to some bond dimension, right?  In which case it's like, well, what's the bomb dimension?

1:44:59 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah. make it infinitely big, then you will just have bigger vectors at the top of your thing, and they're encoding U1 and U2 together, and then U2 and U3 together, and that's not really useful.

1:45:12 - Jacopo Gliozzi
  But I guess if you make it very small, then the hope is that you can preserve some correlation between U1 and U2 without having all about U1 and U2.  So I guess the bond dimension here would be a contraction between these two. Yes, yeah. If these two were contracted, so here there's actually no notion of bond dimension whatsoever.

1:45:37 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  You're right, you're right. So actually, what I'm recreating is exactly what you had drawn before, which is U1 and U2, and then U3.  U3, I guess it's just restricted to the local ones, and so these have some... Right, They have some additional contraction, yeah.  Yeah. Which is why we don't have this local...

1:46:11 - Jacopo Gliozzi
  That like a decent architecture to try. mean, then you would have, yeah, so the question is, yeah, you get these, like, new vectors, right, which are, like, in this scenario, basically all the way up to VT-1 or something.  Yeah, yeah, that's right, that's right, that's right.

1:46:29 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  That's not really what you want, right? You want one, so. I guess the interest, like, one thing that's worth bearing in mind is, like, because the A is a trainable parameter, this still allows you to do the following, which is, like, in theory, you should now get away with much smaller U1s and U2s because you are able to exploit the fact that these can talk to each other and thereby pass information.

1:46:59 - Jacopo Gliozzi
  U1s I'm still a little upset that there's no Bond dimension. I guess you're introducing it here. Yeah, this one's a Bond dimension, right?

1:47:12 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Or also this one. You have two options. Well, it has to be the one that's contracting the neighboring ones, right?  Yeah, but you also have this dimension of the output vectors, which is something you can change, right? Yeah, but the dimension of the output, the way my interpretation of this architecture is like, here is an initial proposal which is purely local.  By virtue of these trainable MPOs, you're making it, let's call it semi-local. And now you have this new semi-local basis, V1 to VT minus 1.

1:47:47 - Jacopo Gliozzi
  In that semi-local basis, that's like a local code, but it encodes some information about the stuff.

1:47:55 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Right, and by the way, I guess however many times you iterate this procedure, I mean, like, determines which kinds of features you're trying to extract, if it's, like, the fully global ones or just the slightly non-local ones or anything in between, right?

1:48:09 - Jacopo Gliozzi
  Oh, nice.

1:48:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So in this case, it's like you have a notion not only of the Bond dimension, which is, like, locally propagating information, but also the, like, hierarchy step, which is, like, how many, how many, like, I guess it's a bit weird, because they're in a sense both, well, yeah, I guess there's the hierarchy step, which is, like, how false you want to make your information, your correlations that you're trying to capture, right?  Yeah, okay, that's interesting. This is, like, you know, I have another project which is trying to do... These hierarchical SAEs, so this is kind of like the two of them together at the same time, which is a bit of a mindfuck to think about.  I guess my only worry is if you just do this one MPS encoding, then what have you captured besides just the most local type of information?  But no, maybe that's what you were saying before. There's a trade-off, and maybe it's a nice cheaper way to do it or something like that.  Yeah, so here's the way I think about it. So let's say we do this, let's say we take the trivial case where we have the same features present at every step.  So it's correlation one between these guys. Then if you just train anything without any temporal correlations, your cost scales as the sequence window that you have.

1:49:49 - Jacopo Gliozzi
  Right.

1:49:49 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  if you're exploiting the correlations, your cost only scales as the like amount of stuff which you're losing by not at you got the of It's Yeah.  Yes, I've got trying trying they're I'm

1:50:00 - Jacopo Gliozzi
  by not capturing the amount of stuff that you're losing, which is not within your contraction. So if this contraction can learn the identity map, then in theory you only need one, but it's a bit confusing because here that one lives separately, right?  Like in my one vector idea, you get that because you're like, oh, I just need to learn one pool of latents that I then decode to every sequence position just trivially, and so I can get away with a much smaller latent space, but in this case, it's a little like, you still have all of the features at each sequence position, and it's not like immediately obvious like how this would work, but if you see what you want in the correlation equals one case.  Then you should average the v's or something like that. Yeah, I think what we need is like some way of like pulling these v's or something.  Yeah. I mean, that's the utility of the tensor network thing, is that you're really compressing some big probability distribution into small pieces because of the local correlations.  Here, I guess, you want a much, like, you know, if you draw this tensor network the way we've drawn it right now in red, it's like, you have these vectors, you 1, you 2, you 2, you 3, which are like all the information that you're starting with, kind of.  And you're not really, like, reducing them that much. You're just like encoding the correlations. So, like, you would need something like the, like, if there's T of them, you need, like, log T of, like, the parameters to store them.  But that's good. That would be good. That would be the ideal, right? That would be, like, you map them to log T parameters.  Maybe you should, I mean, this is very vague and probably not helpful, but you should look into, like, people.  People that are using tensor networks to encode states in tight binding models, it seems like a completely pointless exercise, which is like, they want to do moire, and they want to do really, really big systems.  Right, right, right. They want to do such big systems that then, you know, instead of like, they have their state, which is, let's see, oh sorry, was on the wrong screen.  They have their state, let me go to the right here, which is like, in the moire case is like, you know, the wave function at site one and all the way up to the wave function at site n.  But the problem is that n is like huge. But 10k, yeah. Yeah, and this could also be like an Aubry-Andre model if you want, right?  So then, like, instead of doing that, they say, suppose that like, this n is actually the dimension of some many-body Hilbert space.  Yeah, yeah, yeah. Due to the...

1:53:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I don't know, L, where this is a much smaller number. Then they encode this as an MPS with L sites, so that's like log of N sites, and there's like some nice parallel there because the MPS has like various, it's like somehow encoding different correlations.

1:53:23 - Jacopo Gliozzi
  Like the first, basically like the, yeah, okay, okay, okay, I've said this in a very sketchy way, but now let me say it a little more precisely because I think it can be helpful.  So one is like 0, 0, 0, 0, 0, 0, 1. Then two is like 0, 0, 1, 0, right, in this binary encoding, and this would be like 1, 1, 1, 1, 1, 1, 1, So now like these are like local states or something like that, right, of my Hilbert space, of my many body Hilbert space, which only has, you know, a few bits in it, a few bits, and so the  Different MPS's in code, like, so this, this object that you drew on, phi1 is like a scalar or it's the wave function associated with the first, um, Scalar, scalar, scalar.  Yeah. It's just like, it's just like the value. Yeah. It's like the value of the wave function at site one, i equals one, i equals two.  But now the goal is like, instead of it representing these numbers all the way up to n, we like represent them as like bit strings.  And then this is a much, you know, this Hilbert space is, it's still just as big as my Hilbert space of sites.  Yeah. But I can now compress this because there's like fewer bits into an MPS. So like, you know, with a four, four site fake system, I could get all the way up to, I don't know, what is this, uh, eight plus four plus two plus one sites.  So like 15 sites. So I can represent a 15-site single particle system as a 4-site many-body system, which seems like a stupid thing to do until you then compress the wave function of the many-body system using MPS.  But now, like, the point is, like, because of this bit mapping or something like that, like, so the first half of them have, like, a zero in the first index.  Then the next half of them have, like, a one in the first index, right? Right.

1:55:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So, like, the first bit or the first site of my MPS tells me about correlations between these two halves of my chain.  Then my smallest bit, which is, like, I don't know, my last site of the MPS, tells me about correlations between nearest neighbors on my chain.  Right, right, right, right. So it's a way to kind of encode the correlations at different length scales also, so that it becomes kind of a not translation invariant MPS, because it's likely that things are more correlated, you know, locally than they are globally.

1:55:59 - Jacopo Gliozzi
  But this gives me... Some, I don't know, this is some idea for like something along the lines of what you're trying to do, which is like, just create a synthetic lower dimensional problem, whose MPSs live, give you probability distributions in the larger space that you actually care about.  Yes, I'm not going to lie, that's a bit of a mindfuck, but I'll have to... Yes, it's a very like, I mean, I think numerically it's not so useful unless you do something like very large moiré or something like that.  But that's good because we, I do want a limit, because like, given that our temporal, like naive temporal cross-coder is promising, I think like, it would be very nice if we had a limit where we could go to very large windows, right, very large like sequence lengths, and so if there's some analogue of doing this, that would be good.  I have to think a little bit about that, do you have like a paper in mind that you're... I have to find it, it's by some Portuguese people, I'll send it to you when I find it, but...  me think for two seconds here about your case, right? So suppose that we now have this, this, you know, u1, it's a little bit of a higher dimensional problem, right?  Because the, now the value of u is, uh, is a vector, but let's pretend like it's something. because here you just have the scaler for that wave function.

1:57:20 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah.

1:57:21 - Jacopo Gliozzi
  Yeah. So this is like, let's say we have four times, right? u1, u2, u3, and u4. Yeah. Then like, in some sense, the information that we're coming in from is like, I'm going to vectorize your matrix.  So this is u1, then this is u2, and so on, right? Yep. I wonder if there's some way to do the same kind of representation where like, this is one, which is, you know, 0, 1, this is 2, which is 1, 0, then like 1, 1, okay, something like that.  guess I should have gone from 0, 0, 0, 1. One, zero, and then one, one. These are my one, two, three, four indices.  But now I interpret this as some many-body Hilbert space. So I write this as a super vector or something like that, a vector with on-site dimension, whatever your dimension of your use is.  And then I compress that as an NPS.

1:58:21 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So like, yeah, in the many-body Hilbert space, this is like, I have some U1 here, then I have some like U2, and I'm just rewriting the same vector in this basis.  But then, using this NPS procedure, I can compress it into like an NPS with two sites. So now I'll have two sites, and this is like, you know, local correlations, which captures, like, the correlations between my sites like that, and this site tells me about the global correlations.  same ценist. We've got problems. Okay, IWoo, Which is like these kinds of larger ones. Okay, I'll try to send you that paper, but in principle, this gives you some compression.  I think before, my confusion was that it seems like any architecture we could come up with doesn't reduce the amount of information you're U1, U2, U3, U4, but somehow you do want to reduce it to just capture correlations rather than just keep all of them in a scrambled way.  Well, I don't necessarily want to reduce it. I mean, like, I guess ideally, like, yes, it would be nice to have a, like, just list of features that I could interpret, right?  Just a list of latents that I could interpret, but, like, it's also okay. I think it's also like the way I was thinking about this M, right?  This M was like, I start off with the local codes. I update each local code according to this M, and then I go to, then I decode from there, right?  I think there the intuition was, you know, in a regular...

2:00:00 - Jacopo Gliozzi
  I I have to pack all of the information about correlations between different sequence positions into some local representation, and if I train something which allows the local representations to talk to each other, the corresponding local representations should be much smaller for a given compute budget that I'm training on, or maybe better, for a given level of reconstruction lots.  And so my intuition was that I would still have local code features, but now those local code features would have some of the global information mixed in, which is not ideal from the interpretability standpoint.  But I had some hope that maybe I could then figure out what the local and global contributions of each are by looking at just inner products with the mixing matrix and so on.

2:00:54 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So it's not terrible to have different features at each point, but in the ideal case... Yeah, would have some list of latents where I can say, oh yeah, these are the local features and these are the global ones.

2:01:08 - Jacopo Gliozzi
  But isn't that the beauty of this specific way of doing the tensor network?

2:01:12 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah, I don't really understand it.

2:01:14 - Jacopo Gliozzi
  Your tensor network MPS matrices themselves are like, depending on which one you choose, which site you choose, you're doing global correlation versus global correlation.  Yeah, I found I found so there's a journal club for condensed matter on this, which is basically like, why is this not all ?  And then the guy's like, oh, it's not completely , I guess.

2:01:37 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And then the one of the papers, I think, is this one. So yeah, it's super moiré matter beyond one billion sites.  So there you go. So they're doing moiré of moiré. Little do they know about the HK possibilities in Moiré.  Yeah, yeah, yeah. Well, imagine integrating that with this Moiré of Moiré, and then...

2:02:12 - Jacopo Gliozzi
  Oh, goodness. Thank you for explaining in detail everything about sparse autoencoders and, you know, transformers, which I didn't know.

2:02:26 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  It does seem like there's a lot of, like, tantalizing ways that one should try to use the sorts of entanglement methods of quantum.  And I think my thought is, like, let's just show the V0 of that, right? Like, all I want to do is just show, like, some way in which we understand and in which the tensor network is telling you something that's easy to, like, get a handle on.  And, like, once we do that, if we verify the basic idea, like, us and other people can hill climb on that.  We more complicated things, but it's like, for now, the state of the art is not the best Tensor network implementation, but it's just like no implementation at all.

2:03:11 - Jacopo Gliozzi
  Yeah, yeah. It is cool that just inserting some local temporal correlations into your on thoughts of how to do the architecture, you can get such a marked shift from local to global information.

2:03:25 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  That was a very nice plot, you were right. Yeah, yeah. So, yeah, maybe there's some hope. I mean, there's some part of this which is like, okay, maybe we just do the, like, naivest possible thing of the temporal cross-coder, but I think, like, the paper that we could write, which would be really nice, would be, like, we have the synthetic benchmark, we, like, find the regimes in which each architecture is optimal, and then, like, we do this on language models, we interpret the features that we get, and then we show that, like, the Tensor network allows you to go to, like, super long-range features and, like, we can then, like, interpret some of them, and then be, like, okay, that's very nice, like, people  We need to investigate this technique more, and it's like, job done, you know.

2:04:04 - Jacopo Gliozzi
  This is a cool pipeline, yeah.

2:04:05 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  think that, yeah, I mean, now that you have mentees, you know, this is the real name of the game, is get the mentees to work out all these details about which dimensions should match which.

2:04:20 - Jacopo Gliozzi
  No, but that's...

2:04:21 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But I think, like, with the mentees, right, it's like, you need to do the conceptual work, and then they can do the, like, operational, like, implementation, right, but, like, you do need to, like, they will find it easier insofar as you've done a good job, like, resolving the conceptual clarity.  I think if you're just, like, you have no idea, and you, like, don't, are not doing any of the implementation, I think that's just a recipe for disaster.  But I want to become a meta-ideas guy. I love the meta-ideas to an ideas guy who comes up with the ideas, who then gives them to a calculation guy.  Climb the mirror, another layer of the mirror.

2:05:03 - Jacopo Gliozzi
  That's right, yeah. Meta-ideas are basically like grant proposals. Yeah, basically. It's like meta-ideas is like, why don't we take the SSH chain and think about dipoles in it, you know?  Yes, yeah. I don't think I've reached the level of enlightenment to be proposing good meta-ideas at this point. No, think you need to be like 50 and half the meta-ideas are definitely not worth pursuing.

2:05:36 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah. It's like also you need like some hierarchy, Some time at the implementation level, more time at the ideas level, and then hopefully even more time at the meta-ideas level.  Yeah. All right. Anyway. Yeah, dude, thank you for brainstorming. I feel like this was helpful. I feel like I have like at least a much better handle on like why some of the things are...  It certainly taught me a lot. don't know how productive it was for you, but I think those papers about the tensor network supermore encoding are interesting from a conceptual level about this hierarchy of things, and, you know, it sort of does give you a tuning parameter to understand which correlations you want to choose, but I'm not sure how, yeah, how simple it is to understand.
  ACTION ITEM: Design TN-SAE (W_enc, W_dec, TN); define toy HMM; run sweeps; share w/ Jacopo - WATCH: https://fathom.video/share/Gz5hmVyTjyfwdBytnKt_BBPzCYyaUjos?timestamp=7587.9999  to finish the analogy, let's say. Yeah, yeah. Okay, well, I'll keep thinking about it, dude.

2:06:36 - Jacopo Gliozzi
  Like, I'll keep thinking about it.

2:06:38 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think, like, the, like, pipeline I have for, like, moving these ideas forward is, like, come up with an architecture, write down the encoder-decoder matrices, then, if possible, like, construct, like, an analytic case in which, like, I solve how that architecture performs on a given twin model.

2:06:58 - Jacopo Gliozzi
  Yeah. Yeah, Not just. on that architecture on the toy model using like the four metrics we like developed in our pipeline.

2:07:04 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  And then if it's promising, if we understand like the hyperparameter sweeps, then we like push it to the language model setting.  Uh, so yeah, basically like if I have any like concrete like results on this, I'll, I'll drop you a line and similarly, yeah, I'm curious.  Yeah. And similarly, if you're like, oh yeah, like here are the ways, this the, this is the weight matrix.  Yeah. Yeah. Then I can like, just like either myself tell Claude or tell my mentees to tell Claude to implement it.  Okay. I'll, I'll, I'll keep thinking about it then in the back burner if I can give you a concrete set of weight matrices to.  Yeah, yeah, concrete set of weight matrices and also like the toy problem, right?

2:07:46 - Jacopo Gliozzi
  If you're like, oh, this is the like hidden Markov model that would opt, that would optimally be, optimally be captured by this architecture, that's also super helpful.  Yeah. Right.

2:07:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  You want to induce some correlations, but what, what about the one that you. It's fine, but there are some issues, like in the one that we do, we don't have a good hand-along complexity, so it's basically like a single latent is associated with a single firing, a single feature.

2:08:14 - Jacopo Gliozzi
  There are generalizations where you can do more features, or a single latent drives many different features, and so on.  And it's just a case of like, I assume that for a given architecture, a given assumption about the HMM is probably optimal.  And so, yeah, there's some work there about like, how do we understand the scaling of the HMM with the scaling of the reconstruction architecture?

2:08:45 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Anyway, anyway, anyway. Thank you for also taking the time to explain so much of this stuff.

2:08:51 - Jacopo Gliozzi
  mean, that's... Yeah, yeah, no, of course.

2:08:53 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  There's nothing, and now I know something, but we'll see. And the key thing is... Not a lot to know.

2:09:07 - Jacopo Gliozzi
  Yeah, what are you defending, by the way? Ideally, in July.

2:09:11 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  mean, ideally, I have to do in July. I got buried to say he would be on my committee.

2:09:17 - Jacopo Gliozzi
  I have to replace Smita, who's going to be in India, and also Jake Coby, who's moving to Chicago. I did, so chaos and UIEC, as always.  Yeah, I need to replace him soon, because the month of May is the month I'm going to be writing my thesis.

2:09:35 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So the month of April is the month I need to finish the projects that I need to finish. Is Taylor, like, he's good with you graduating, or he doesn't know at this point?

2:09:48 - Jacopo Gliozzi
  No, no, he knows, he knows, he knows. He's the one who said do it in July. Okay, thank you.  Okay, huge. right, yeah, Chief, thank you for chatting. Good to know the... You know, things are more or less where we're lost.  And I guess like you'll move in July as well, right? Yeah, probably like the end of July, like the very end of it.  Okay, okay, okay. Well, dude, it'll be an epochal moment for our graduate career. We won't think of it yet.  We won't think of it yet, yeah, that's true. Hopefully you'll be back in Urbana for at least a couple days or I'll be in.  I mean, once I move, I'll try to travel to Berkeley at some point during my postdoc, because I have travel money and there's lots of things in Berkeley, so.  Yeah, yeah, Rekindle your relationship with Effort. Yeah, my non-existent relationship, but yeah. Cool. All right, Chief then, nice to see you.  As always, good luck with the various pipelines you've put in motion. All right, Chief then, safe. Bye.