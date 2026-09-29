# FrameFT, explained: narration script

Generated from `narration.py`; this is the text the voiceover reads and the captions show.

## 01 · FrameFT, end to end

This is RoBERTa, a language model with about 125 million learned numbers. To teach it a new task, the obvious move is to adjust all of them. FrameFT adjusts 24,000. That's 0.02% of the model. And on language benchmarks, it matches, or even beats, full fine-tuning.

In this video, we'll walk through the whole process: the problem FrameFT solves, what a frame actually is, how the frame gets built, how training works, and why the code makes the choices it does. Then, an experiment we're running to test whether the frame itself is what makes it work.

## 02 · The problem: fine-tuning

Most of what a model knows lives in weight matrices: big grids of numbers. A layer takes an input vector, multiplies it by its matrix, and produces an output vector. In RoBERTa-base, a typical matrix is 768 by 768, which is almost 600,000 numbers.

Fine-tuning means learning a change to each matrix. We'll call that change ΔW, where delta just means "change in". The new weights are the old weights plus ΔW. The old weights stay frozen, and all of the learning goes into ΔW.

If ΔW can be any matrix at all, we're back to training hundreds of thousands of numbers per matrix, and storing a whole new copy of the model for every task. So the real question is: how can we describe ΔW with far fewer numbers?

## 03 · Parameter-efficient fine-tuning

Methods that answer this are called parameter-efficient fine-tuning, or PEFT. The best known is LoRA. It writes ΔW as a tall, thin matrix times a short, wide one. The thin side is called the rank: roughly, how many independent patterns ΔW can contain. At rank eight, that's about 12,000 numbers per matrix, instead of 600,000.

There's another way to be economical. Choose a fixed dictionary of patterns ahead of time, and learn only how much of each pattern to use. Those amounts are called coefficients. It's the same trick JPEG uses to compress a photo: it stores the amounts of simple wave patterns, instead of every pixel.

FourierFT, one of the paper's baselines, uses Fourier waves as its dictionary. FrameFT uses something more general, called a tight fusion frame. To see what that is, we'll build up three ideas: a basis, a frame, and a fusion frame.

## 04 · Bases and frames

Start with a basis. In a flat plane, two arrows at right angles, each of length one, form an orthonormal basis. Any vector is some amount of the first arrow plus some amount of the second, and those amounts are just its shadows on each arrow, also called dot products. A nice bonus: the squared shadows add up to the squared length of the vector. Nothing is lost.

A frame relaxes one rule: you may use more arrows than dimensions. Here are three arrows, evenly spaced, in a two-dimensional plane. Every vector can still be built from them. There's just more than one way to do it, and that extra slack is called redundancy.

A frame is tight if it treats every direction equally. Watch the three squared shadows as the vector spins around. Each one rises and falls, but their sum never changes: it's always exactly one and a half times the squared length. Now compare a lopsided frame, with the arrows bunched together. The total wobbles, because some directions get more attention than others.

Tightness buys something practical. To rebuild a vector, measure its shadows, then add the arrows back, weighted by those shadows, and divide by that constant. Measure, then rebuild. Frame theory calls these two steps analysis and synthesis, and both will show up again inside FrameFT.

## 05 · Fusion frames

A fusion frame goes one step further. Instead of single arrows, its building blocks are whole subspaces, like lines and planes through the origin. And instead of a shadow on an arrow, you take the shadow on each subspace, called a projection.

The fusion frame is tight when the squared lengths of all the projections always add up to the same multiple of the vector's squared length. Here, a plane and a perpendicular line split three-dimensional space. The two shadows obey Pythagoras: together they account for the whole vector, however it points.

This is exactly the kind of frame FrameFT's code builds. The redundancy is one, so the subspaces are perpendicular and fit together with no overlap. For RoBERTa, the 768-dimensional space is carved into 384 perpendicular planes. Stack the frame's vectors as rows, two per plane, and you get a 768 by 768 matrix called B. It's an orthogonal matrix: its rows are perpendicular and of length one, so it can rotate space, but never stretches it.

## 06 · Building the frame

How is B built? A key design choice: B is never trained, and never saved. It's regenerated from just two numbers, the size n and the subspace dimension l, and it comes out identical every time. So one copy can be shared by every layer of that size.

The construction has three steps. Step one is called Spectral Tetris. It builds a small seed matrix whose columns all have length one, and whose rows are perpendicular and carry equal energy. It fills each row with ones, from left to right. When a row needs a fractional amount to finish, it drops in a two by two block, the tetris piece, which splits the leftover perfectly between two rows. In FrameFT's settings the sizes always divide evenly, so the seed is simply a staircase of ones.

Step two is modulation, and it uses complex numbers, which you can picture as arrows, where multiplying means rotating. Make many copies of the seed. In each copy, rotate every column by an angle that grows steadily along the row, and give every copy its own speed, like clock hands spinning at different rates. Compare two copies with different speeds, and the rotations cancel out, so the copies are perpendicular. It's the same cancellation that makes the Fourier transform work.

Step three: neural networks use real numbers, not complex ones. So each complex number, a plus i b, is replaced by a two by two block of real numbers that rotates and scales in exactly the same way. This doubles the size, turning 384 complex coordinates into 768 real ones.

Here's the result for RoBERTa: a 768 by 768 orthogonal matrix whose rows are cosine and sine waves of increasing frequency. At these settings, the tight fusion frame is a real-valued Fourier basis.

## 07 · The FrameFT update

Now the main idea. FrameFT writes the update as B transpose, times S, times B, times a scale factor. B is the frozen frame. S is a 768 by 768 grid that's almost entirely zeros. Only a thousand of its entries may be nonzero, and those thousand numbers are the only things FrameFT trains for this matrix.

Read it from the input's point of view. First, B transpose measures the input's shadows on every frame direction. That's analysis. Then S edits those measurements, scaling some, and mixing a few into others. Finally, B turns the edited measurements back into an ordinary vector. That's synthesis. Measure, edit, rebuild.

Another way to see it: each coefficient owns one fixed pattern, called an outer product: one frame row stood on its end, times another lying flat. Frame rows are waves spread across the whole vector, so each pattern covers the entire matrix. A single coefficient nudges nearly all 600,000 weights at once, in a coordinated wave, and ΔW is simply the sum of a thousand such patterns.

Where do the thousand positions go? They're picked at random, once, using a seed: the starting number for a random number generator, so the same seed always makes the same choices. Because the seed can regenerate them, the positions never need to be saved. And with the share-entry option used here, every layer uses the same positions.

## 08 · Two knobs: block size and scale

Two knobs shape S. The first is the block size. With a block size of 768, used for MRPC, a coefficient can sit anywhere, linking any frame direction to any other. With a block size of two, used for RTE, coefficients may only sit in the two by two blocks along the diagonal. Each block links one plane to itself, so each frequency is adjusted on its own, like the sliders on an audio equalizer. That's the block-diagonal structure drawn in the paper.

The second knob is the scale. The code divides by n, then multiplies by a tuned scale: ten for RTE, fifty for MRPC. Because B never stretches anything, the size of ΔW is exactly scale over n, times the size of the coefficient vector. So the scale sets how hard the coefficients push, and it trades off against the learning rate, the step size of each training update. Learning rate times scale comes out similar for both tasks: between three and four.

One more detail. In the GLUE runs, the coefficients start as random numbers, not at zero like LoRA, so training begins from a small random nudge.

## 09 · Training and inference

Now into the real model. RoBERTa-base has twelve layers. In each one, FrameFT wraps two matrices in the attention block: the query and value projections. Roughly, the query decides what each word looks for in the other words, and the value decides what information it passes along. Everything original is frozen. The only trainable pieces are the thousand coefficients per wrapped matrix, plus a small classification head on top.

Each time the model reads an input, called the forward pass, the layer computes its usual output and adds the input times ΔW, rebuilt from the current coefficients. On the backward pass, the gradient arrives: for every weight, which way it should move to reduce the error. Each coefficient gets its share: the part of that gradient that lines up with its own pattern. The optimizer uses two learning rates: a large one for the coefficients, which are heavily scaled down, and a smaller one for the head.

Two engineering choices keep this cheap. Each basis is built once per size and cached, with one copy per GPU, shared by every layer. And after training, ΔW can be merged straight into W, so the fine-tuned model runs exactly as fast as the original.

Count it up: twelve layers, times two matrices, times a thousand coefficients, is 24,000 numbers. LoRA, at rank eight, needs about 300,000. Full fine-tuning needs 125 million. Stored as 32-bit numbers, a FrameFT adapter is about 96 kilobytes.

## 10 · Results from the paper

Does it work? The paper tests on GLUE, a standard set of language-understanding tasks, like deciding whether one sentence implies another, or whether two sentences say the same thing. On RoBERTa-base, FrameFT averages 86.1, compared with 85.2 for both LoRA and full fine-tuning, while training 24,000 numbers instead of 300,000 or 125 million.

The pattern holds when instruction-tuning Llama 2, with 7 billion parameters, and on vision transformers, where FrameFT beats LoRA and FourierFT while training fewer numbers than either.

## 11 · Our experiment: does the frame matter?

Now, the experiment we're running. At these settings, the frame is just an orthogonal matrix. So does its wave structure matter, or would any orthogonal basis do? We swap B for two alternatives, and change nothing else.

The random arm uses a uniformly random rotation, made by taking a grid of random numbers and straightening it into an orthogonal matrix with a QR decomposition. The identity arm uses the plainest basis of all, where each coefficient changes exactly one weight.

To keep it fair, all three share the same coefficient positions, seeds, data order, and settings. And orthogonality gives a neat guarantee: with the same coefficients, all three updates have exactly the same size, rank, and singular values, which measure how strongly a matrix stretches along its main directions. They differ only in orientation: which input directions they read, and which output directions they write.

Results so far. On RTE, with three seeds each, meaning three repeat runs with different randomness, frame and random are neck and neck at their best epoch, around 79.5, right at the paper's number, while identity trails by seven points. At the final epoch, frame leads random by about two points. On MRPC, with fewer seeds, the order is frame, then random, then identity. And all three reach similar training loss, so identity isn't failing to learn. It learns something that generalizes worse.

One likely reason: with a thousand single entries scattered over the matrix, about a quarter of the input features never touch the update at all. Dense bases give every coefficient a say over the whole matrix. So spreading coefficients out clearly matters. Whether the frame's particular waves beat a random rotation is a smaller effect. With two or three seeds, and a validation set where one example is worth a third of a point, it needs more seeds to settle.

## 12 · Recap

Let's recap the whole pipeline. From two numbers, n and l, Spectral Tetris, modulation, and a real-valued rewrite build a frozen, orthogonal frame B, shared by every layer of that size. From a seed, a thousand positions are picked in a sparse matrix S, and only those coefficients are trained. The update is scale over n, times B transpose, S, B: measure, edit, rebuild. It's added to the frozen weights, and merged in when training is done.

Twenty-four thousand numbers, a couple of seeds, and a formula. That's the whole adapter. Thanks for watching.

