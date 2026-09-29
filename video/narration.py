"""The voiceover script for the FrameFT explainer video.

Every paragraph is one narration segment; the scene that shows it calls `self.narrate("<id>")`.

Markup inside the text:
    {display|spoken}   shown one way in the captions, read another way by the voice
    [[mark]]           a bookmark; scenes can wait for the exact moment this word is spoken

Words in SPOKEN_REPLACEMENTS are respelled for the voice only, so the captions keep the real names.
"""
import re

SPOKEN_REPLACEMENTS = [
    (r"\bFrameFT\b", "Frame F T"),
    (r"\bFourierFT\b", "Fourier F T"),
    (r"ΔW", "delta W"),
    (r"\bLoRA\b", "Lora"),
    (r"\bRoBERTa-base\b", "Roberta base"),
    (r"\bRoBERTa\b", "Roberta"),
    (r"\bPEFT\b", "P E F T"),
    (r"\bRTE\b", "R T E"),
    (r"\bMRPC\b", "M R P C"),
    (r"\bGLUE\b", "Glue"),
    (r"\bQR\b", "Q R"),
    (r"\bGPU\b", "G P U"),
    (r"\bJPEG\b", "jay peg"),
]

SCRIPT = {
    # ------------------------------------------------------------------ 01
    "intro": [
        "This is [[model]]RoBERTa, a language model with about [[count]]{125 million|one hundred twenty-five million} "
        "learned numbers. To teach it a new task, the obvious move is to adjust [[all]]all of them. "
        "FrameFT adjusts [[few]]{24,000|twenty-four thousand}. That's [[pct]]{0.02%|two hundredths of one percent} of the model. "
        "And on language benchmarks, it [[match]]matches, or even beats, full fine-tuning.",

        "In this video, we'll walk through the whole process: [[c1]]the problem FrameFT solves, "
        "[[c2]]what a frame actually is, [[c3]]how the frame gets built, [[c4]]how training works, and why the code "
        "makes the choices it does. Then, [[c5]]an experiment we're running to test whether the frame itself is what "
        "makes it work.",
    ],
    # ------------------------------------------------------------------ 02
    "problem": [
        "Most of what a model knows lives in [[mats]]weight matrices: big grids of numbers. A layer takes an "
        "[[x]]input vector, multiplies it by its [[W]]matrix, and produces an [[y]]output vector. In RoBERTa-base, a "
        "typical matrix is [[size]]{768 by 768|seven sixty-eight by seven sixty-eight}, which is almost "
        "{600,000|six hundred thousand} numbers.",

        "Fine-tuning means learning a [[dw]]change to each matrix. We'll call that change ΔW, where delta just means "
        "\"change in\". The new weights are the [[sum]]old weights plus ΔW. The old weights stay [[frozen]]frozen, and "
        "all of the learning goes into ΔW.",

        "If ΔW can be [[any]]any matrix at all, we're back to training hundreds of thousands of numbers per matrix, "
        "and [[copies]]storing a whole new copy of the model for every task. So the real question is: [[q]]how can we "
        "describe ΔW with far fewer numbers?",
    ],
    # ------------------------------------------------------------------ 03
    "landscape": [
        "Methods that answer this are called [[peft]]parameter-efficient fine-tuning, or PEFT. The best known is "
        "[[lora]]LoRA. It writes ΔW as a [[tall]]tall, thin matrix times a [[wide]]short, wide one. The thin side is "
        "called the [[rank]]rank: roughly, how many independent patterns ΔW can contain. At rank eight, that's about "
        "[[count]]{12,000|twelve thousand} numbers per matrix, instead of {600,000|six hundred thousand}.",

        "There's another way to be economical. Choose a [[dict]]fixed dictionary of patterns ahead of time, and learn "
        "only [[coef]]how much of each pattern to use. Those amounts are called coefficients. It's the same trick "
        "[[jpeg]]JPEG uses to compress a photo: it stores the amounts of simple wave patterns, instead of every pixel.",

        "[[fourier]]FourierFT, one of the paper's baselines, uses Fourier waves as its dictionary. [[frameft]]FrameFT "
        "uses something more general, called a [[tff]]tight fusion frame. To see what that is, we'll build up three "
        "ideas: [[b]]a basis, [[f]]a frame, and [[ff]]a fusion frame.",
    ],
    # ------------------------------------------------------------------ 04
    "frames": [
        "Start with a [[basis]]basis. In a flat plane, two arrows at right angles, each of length one, form an "
        "[[ortho]]orthonormal basis. Any vector is some amount of the first arrow plus some amount of the second, and "
        "those amounts are just its [[shadows]]shadows on each arrow, also called dot products. A nice bonus: the "
        "[[pyth]]squared shadows add up to the squared length of the vector. Nothing is lost.",

        "A [[frame]]frame relaxes one rule: you may use more arrows than dimensions. Here are [[three]]three arrows, "
        "evenly spaced, in a two-dimensional plane. Every vector can still be built from them. There's just "
        "[[redund]]more than one way to do it, and that extra slack is called redundancy.",

        "A frame is [[tight]]tight if it treats every direction equally. Watch the three squared shadows as the "
        "vector [[spin]]spins around. Each one rises and falls, but their [[sum]]sum never changes: it's always exactly "
        "one and a half times the squared length. Now compare a [[lopsided]]lopsided frame, with the arrows bunched "
        "together. The total [[wobble]]wobbles, because some directions get more attention than others.",

        "Tightness buys something practical. To rebuild a vector, [[measure]]measure its shadows, then "
        "[[rebuild]]add the arrows back, weighted by those shadows, and divide by that constant. Measure, then "
        "rebuild. Frame theory calls these two steps [[names]]analysis and synthesis, and both will show up again "
        "inside FrameFT.",
    ],
    # ------------------------------------------------------------------ 05
    "fusion": [
        "A [[fusion]]fusion frame goes one step further. Instead of single arrows, its building blocks are whole "
        "[[subspaces]]subspaces, like lines and planes through the origin. And instead of a shadow on an arrow, you "
        "take the shadow on each subspace, called a [[proj]]projection.",

        "The fusion frame is [[tightf]]tight when the squared lengths of all the projections always add up to the "
        "same multiple of the vector's squared length. Here, a [[plane]]plane and a [[line]]perpendicular line split "
        "three-dimensional space. The two shadows obey [[pyth]]Pythagoras: together they account for the whole "
        "vector, however it points.",

        "This is exactly the kind of frame FrameFT's code builds. The [[red1]]redundancy is one, so the subspaces are "
        "perpendicular and fit together with no overlap. For RoBERTa, the [[carve]]{768|seven hundred sixty-eight}-"
        "dimensional space is carved into [[planes]]{384|three hundred eighty-four} perpendicular planes. Stack the "
        "frame's vectors as [[rows]]rows, two per plane, and you get a [[B]]{768 by 768|seven sixty-eight by seven "
        "sixty-eight} matrix called B. It's an [[orth]]orthogonal matrix: its rows are perpendicular and of length "
        "one, so it can rotate space, but never stretches it.",
    ],
    # ------------------------------------------------------------------ 06
    "build": [
        "How is B built? A key design choice: B is [[never]]never trained, and never saved. It's "
        "[[regen]]regenerated from just two numbers, the size n and the subspace dimension l, and it comes out "
        "identical every time. So [[share]]one copy can be shared by every layer of that size.",

        "The construction has three steps. Step one is called [[tetris]]Spectral Tetris. It builds a small seed "
        "matrix whose [[cols]]columns all have length one, and whose [[rows]]rows are perpendicular and carry equal "
        "energy. It fills each row with ones, from left to right. [[piece]]When a row needs a fractional amount to "
        "finish, it drops in a two by two block, the tetris piece, which splits the leftover perfectly between two "
        "rows. [[even]]In FrameFT's settings the sizes always divide evenly, so the seed is simply a staircase of ones.",

        "Step two is [[mod]]modulation, and it uses complex numbers, which you can picture as [[arrows]]arrows, where "
        "multiplying means rotating. Make many copies of the seed. In each copy, [[rotate]]rotate every column by an "
        "angle that grows steadily along the row, and give every copy its own speed, like [[clocks]]clock hands "
        "spinning at different rates. Compare two copies with different speeds, and the rotations [[cancel]]cancel "
        "out, so the copies are perpendicular. It's the same cancellation that makes the Fourier transform work.",

        "Step three: neural networks use real numbers, not complex ones. So each complex number, [[complex]]a plus i "
        "b, is replaced by a [[block]]two by two block of real numbers that rotates and scales in exactly the same "
        "way. This [[double]]doubles the size, turning {384|three hundred eighty-four} complex coordinates into "
        "{768|seven hundred sixty-eight} real ones.",

        "Here's the [[result]]result for RoBERTa: a {768 by 768|seven sixty-eight by seven sixty-eight} orthogonal "
        "matrix whose rows are [[waves]]cosine and sine waves of increasing frequency. At these settings, the tight "
        "fusion frame is a [[fourier]]real-valued Fourier basis.",
    ],
    # ------------------------------------------------------------------ 07
    "update": [
        "Now the main idea. FrameFT writes the update as [[bt]]B transpose, times [[S]]S, times [[B]]B, times a "
        "[[scale]]scale factor. B is the frozen frame. [[Sgrid]]S is a {768 by 768|seven sixty-eight by seven "
        "sixty-eight} grid that's almost entirely zeros. Only [[thousand]]a thousand of its entries may be nonzero, "
        "and those thousand numbers are the [[only]]only things FrameFT trains for this matrix.",

        "Read it from the input's point of view. First, B transpose [[analysis]]measures the input's shadows on "
        "every frame direction. That's analysis. Then S [[edit]]edits those measurements, scaling some, and mixing a "
        "few into others. Finally, B [[synth]]turns the edited measurements back into an ordinary vector. That's "
        "synthesis. [[mer]]Measure, edit, rebuild.",

        "Another way to see it: each coefficient owns [[atom]]one fixed pattern, called an outer product: one frame "
        "row stood on its end, times another lying flat. Frame rows are waves spread across the whole vector, so "
        "[[covers]]each pattern covers the entire matrix. A single coefficient [[nudge]]nudges nearly all "
        "{600,000|six hundred thousand} weights at once, in a coordinated wave, and ΔW is simply the "
        "[[sumatoms]]sum of a thousand such patterns.",

        "Where do the thousand positions go? They're [[random]]picked at random, once, using a seed: the starting "
        "number for a random number generator, so the same seed always makes the same choices. Because the seed can "
        "regenerate them, the positions [[nosave]]never need to be saved. And with the [[share]]share-entry option "
        "used here, every layer uses the same positions.",
    ],
    # ------------------------------------------------------------------ 08
    "knobs": [
        "Two knobs shape S. The first is the [[bs]]block size. With a block size of {768|seven sixty-eight}, "
        "[[mrpc]]used for MRPC, a coefficient can sit anywhere, linking any frame direction to any other. With a "
        "block size of two, [[rte]]used for RTE, coefficients may only sit in the [[diag]]two by two blocks along "
        "the diagonal. Each block links one plane to itself, so [[eq]]each frequency is adjusted on its own, like the "
        "sliders on an audio equalizer. That's the [[paper]]block-diagonal structure drawn in the paper.",

        "The second knob is the [[scale]]scale. The code divides by n, then multiplies by a tuned scale: "
        "[[vals]]ten for RTE, fifty for MRPC. Because B never stretches anything, the size of ΔW is [[norm]]exactly "
        "scale over n, times the size of the coefficient vector. So the scale sets how hard the coefficients push, "
        "and it trades off against the learning rate, the step size of each training update. [[prod]]Learning rate "
        "times scale comes out similar for both tasks: between three and four.",

        "One more detail. In the GLUE runs, the coefficients [[init]]start as random numbers, not at zero like "
        "LoRA, so training begins from a small random nudge.",
    ],
    # ------------------------------------------------------------------ 09
    "training": [
        "Now into the real model. RoBERTa-base has [[layers]]twelve layers. In each one, FrameFT wraps two matrices "
        "in the attention block: the [[qv]]query and value projections. Roughly, the query decides [[query]]what each "
        "word looks for in the other words, and the value decides [[value]]what information it passes along. "
        "Everything original is [[frozen]]frozen. The only trainable pieces are the [[coeffs]]thousand coefficients "
        "per wrapped matrix, plus a small [[head]]classification head on top.",

        "Each time the model reads an input, called the forward pass, the layer computes its [[fwd]]usual output and "
        "adds [[plus]]the input times ΔW, rebuilt from the current coefficients. On the backward pass, the "
        "[[grad]]gradient arrives: for every weight, which way it should move to reduce the error. [[gshare]]Each "
        "coefficient gets its share: the part of that gradient that lines up with its own pattern. The optimizer "
        "uses [[lrs]]two learning rates: a large one for the coefficients, which are heavily scaled down, and a "
        "smaller one for the head.",

        "Two engineering choices keep this cheap. Each basis is [[cache]]built once per size and cached, with one "
        "copy per GPU, shared by every layer. And after training, ΔW can be [[merge]]merged straight into W, so the "
        "fine-tuned model runs [[fast]]exactly as fast as the original.",

        "Count it up: [[count]]twelve layers, times two matrices, times a thousand coefficients, is "
        "{24,000|twenty-four thousand} numbers. [[lora]]LoRA, at rank eight, needs about {300,000|three hundred "
        "thousand}. [[ff]]Full fine-tuning needs {125 million|one hundred twenty-five million}. Stored as "
        "{32-bit|thirty-two bit} numbers, a FrameFT adapter is about [[kb]]{96 kilobytes|ninety-six kilobytes}.",
    ],
    # ------------------------------------------------------------------ 10
    "results": [
        "Does it work? The paper tests on [[glue]]GLUE, a standard set of language-understanding tasks, like deciding "
        "whether one sentence implies another, or whether two sentences say the same thing. On RoBERTa-base, "
        "[[avg]]FrameFT averages {86.1|eighty-six point one}, compared with {85.2|eighty-five point two} for both LoRA "
        "and full fine-tuning, [[params]]while training {24,000|twenty-four thousand} numbers instead of "
        "{300,000|three hundred thousand} or {125 million|one hundred twenty-five million}.",

        "The pattern holds when [[llama]]instruction-tuning {Llama 2|Llama two}, with {7 billion|seven billion} "
        "parameters, and on [[vit]]vision transformers, where FrameFT beats LoRA and FourierFT while training fewer "
        "numbers than either.",
    ],
    # ------------------------------------------------------------------ 11
    "experiment": [
        "Now, the experiment we're running. At these settings, the frame is [[orth]]just an orthogonal matrix. So "
        "does its wave structure matter, or would [[any]]any orthogonal basis do? We swap B for two alternatives, and "
        "change nothing else.",

        "The [[random]]random arm uses a uniformly random rotation, made by taking a grid of random numbers and "
        "straightening it into an orthogonal matrix with a [[qr]]QR decomposition. The [[identity]]identity arm uses "
        "the plainest basis of all, where [[one]]each coefficient changes exactly one weight.",

        "To keep it fair, all three share the [[same]]same coefficient positions, seeds, data order, and settings. "
        "And orthogonality gives a neat guarantee: with the same coefficients, all three updates have [[sv]]exactly "
        "the same size, rank, and singular values, which measure how strongly a matrix stretches along its main "
        "directions. They differ only in [[orient]]orientation: which input directions they read, and which output "
        "directions they write.",

        "Results so far. On [[rte]]RTE, with three seeds each, meaning three repeat runs with different randomness, "
        "frame and random are [[neck]]neck and neck at their best epoch, around {79.5|seventy-nine and a half}, "
        "[[paperline]]right at the paper's number, while [[idgap]]identity trails by seven points. At the "
        "[[final]]final epoch, frame leads random by about two points. On [[mrpc]]MRPC, with fewer seeds, the order is "
        "frame, then random, then identity. And all three reach [[loss]]similar training loss, so identity isn't "
        "failing to learn. It learns something that generalizes worse.",

        "One likely reason: with a thousand single entries scattered over the matrix, [[quarter]]about a quarter of "
        "the input features never touch the update at all. Dense bases give [[dense]]every coefficient a say over the "
        "whole matrix. So spreading coefficients out clearly matters. Whether the frame's particular waves "
        "[[beat]]beat a random rotation is a smaller effect. With two or three seeds, and a validation set where "
        "[[oneex]]one example is worth a third of a point, it needs more seeds to settle.",
    ],
    # ------------------------------------------------------------------ 12
    "recap": [
        "Let's recap the whole pipeline. From two numbers, n and l, [[build]]Spectral Tetris, modulation, and a "
        "real-valued rewrite build a frozen, orthogonal frame B, shared by every layer of that size. From a seed, "
        "[[S]]a thousand positions are picked in a sparse matrix S, and only those coefficients are trained. The "
        "update is [[eq]]scale over n, times B transpose, S, B: measure, edit, rebuild. It's [[add]]added to the "
        "frozen weights, and [[merge]]merged in when training is done.",

        "[[end]]Twenty-four thousand numbers, a couple of seeds, and a formula. That's the whole adapter. "
        "[[thanks]]Thanks for watching.",
    ],
}

CHAPTERS = [
    ("intro", "FrameFT, end to end"),
    ("problem", "The problem: fine-tuning"),
    ("landscape", "Parameter-efficient fine-tuning"),
    ("frames", "Bases and frames"),
    ("fusion", "Fusion frames"),
    ("build", "Building the frame"),
    ("update", "The FrameFT update"),
    ("knobs", "Two knobs: block size and scale"),
    ("training", "Training and inference"),
    ("results", "Results from the paper"),
    ("experiment", "Our experiment: does the frame matter?"),
    ("recap", "Recap"),
]

_MARK = re.compile(r"\[\[(\w+)\]\]")
_BRACE = re.compile(r"\{([^{}|]*)\|([^{}]*)\}")


def display_text(markup):
    """The caption form: braces resolved to their display half, bookmarks removed."""
    return _BRACE.sub(lambda m: m.group(1), _MARK.sub("", markup))


def spoken_text(markup):
    """The voice form: braces resolved to their spoken half, replacements applied, bookmarks kept."""
    parts, pos = [], 0
    for m in _BRACE.finditer(markup):
        parts.append(_respell(markup[pos:m.start()]))
        parts.append(m.group(2))
        pos = m.end()
    parts.append(_respell(markup[pos:]))
    return "".join(parts)


def _respell(text):
    for pattern, repl in SPOKEN_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    return text


def split_sentences(markup):
    """Split a paragraph into sentences, never inside {…} braces."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(markup):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0 and ch in ".?!" and (i + 1 == len(markup) or markup[i + 1] == " "):
            # keep decimals like 86.1 and abbreviations inside braces intact
            out.append(markup[start:i + 1].strip())
            start = i + 1
    rest = markup[start:].strip()
    if rest:
        out.append(rest)
    return [s for s in out if display_text(s).strip()]


def segments():
    """Yield (segment_id, paragraph_markup) in video order, e.g. ("intro_0", "...")."""
    for chapter, _ in CHAPTERS:
        for i, para in enumerate(SCRIPT[chapter]):
            yield f"{chapter}_{i}", para


def markdown():
    """The script as a readable document (what the captions show)."""
    lines = ["# FrameFT, explained: narration script", "",
             "Generated from `narration.py`; this is the text the voiceover reads and the captions show.", ""]
    titles = dict(CHAPTERS)
    for i, (chapter, _) in enumerate(CHAPTERS):
        lines += [f"## {i + 1:02d} · {titles[chapter]}", ""]
        for para in SCRIPT[chapter]:
            lines += [display_text(para).strip(), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--markdown" in sys.argv:
        print(markdown())
        raise SystemExit
    words = 0
    for sid, para in segments():
        n = len(display_text(para).split())
        words += n
        print(f"{sid:14s} {n:4d} words")
    print(f"total {words} words, about {words / 143:.1f} min at 143 wpm")
