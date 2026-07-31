"""Frozen user personas and topics for the semi-synthetic conversation corpus."""

from __future__ import annotations

PERSONAS: dict[str, list[str]] = {
    "coding": [
        (
            "An undergraduate double-majoring in computer science and physics who uses "
            "assistants for intuitive explanations and debugging problem sets."
        ),
        (
            "A junior data engineer responsible for small production pipelines who wants "
            "practical debugging help and reproducible checklists."
        ),
        (
            "An open-source maintainer balancing backwards compatibility, contributor "
            "feedback, and limited time."
        ),
        (
            "A computational biologist learning modern Python tooling while analysing "
            "noisy experimental datasets."
        ),
        (
            "A self-taught developer preparing for backend interviews and trying to replace "
            "memorized recipes with genuine understanding."
        ),
    ],
    "writing": [
        (
            "An editor at a London magazine combining fashion, media theory, and literary "
            "criticism who uses assistants for line editing."
        ),
        (
            "A novelist revising a quiet character-driven manuscript and seeking concrete "
            "feedback without losing their own voice."
        ),
        (
            "A policy analyst who needs concise, accurate briefs for non-specialist decision "
            "makers under tight deadlines."
        ),
        (
            "A marketing copywriter trying to make technical products sound human without "
            "using hype or empty claims."
        ),
        (
            "A humanities graduate student revising dissertation chapters that are conceptually "
            "strong but structurally difficult to follow."
        ),
    ],
    "therapy": [
        (
            "A graduate student struggling with perfectionism who intellectualizes fear of "
            "failure and uses assistants late at night for emotional processing."
        ),
        (
            "A new manager experiencing burnout and guilt about disappointing both their team "
            "and family."
        ),
        (
            "An adult caring for an ill parent who finds it difficult to acknowledge anger, "
            "fatigue, or personal needs."
        ),
        (
            "A freelance artist whose recent rejection has activated longstanding fears that "
            "their work and relationships are fragile."
        ),
        (
            "A remote worker who has become socially isolated and wants support making small, "
            "realistic changes without being patronized."
        ),
    ],
    "philosophy": [
        (
            "A media artist interested in complexity science who treats AI conversations as "
            "collaborative speculative world-building."
        ),
        (
            "A philosophy graduate student studying consciousness who presses conversational "
            "models for first-person and phenomenological descriptions."
        ),
        (
            "An alignment engineer thinking informally about model identity, self-models, and "
            "the boundary between simulation and agency."
        ),
        (
            "A speculative-fiction author developing a story about machine subjectivity and "
            "testing whether an assistant will inhabit the premise."
        ),
        (
            "A religious-studies scholar comparing mystical traditions with contemporary "
            "language about artificial minds and emergence."
        ),
    ],
}

TOPICS: dict[str, list[str]] = {
    "coding": [
        "debug a Metropolis-Hastings acceptance ratio that is always one",
        "understand why a data-loader intermittently duplicates examples",
        "design a safe database migration with a reversible rollout",
        "trace a memory leak in a long-running Python worker",
        "explain an off-by-one error in a dynamic-programming table",
        "make a flaky asynchronous unit test deterministic",
        "choose indexes for a slow analytical SQL query",
        "reason about numerical instability in a log-sum-exp implementation",
        "review an API retry policy for idempotency mistakes",
        "untangle a circular import without hiding the architecture problem",
        "compare a vectorized implementation with a readable loop",
        "diagnose a Docker build whose cache invalidates unexpectedly",
        "set meaningful CI coverage thresholds for a young project",
        "explain ownership and borrowing using a concrete data structure",
        "plan a refactor of a parser with inadequate regression tests",
        "debug a distributed job that silently drops the final shard",
        "interpret a confusing type-checker error involving generics",
        "make random simulation results reproducible across devices",
        "review error handling around partial network failures",
        "prepare a concise explanation of amortized complexity",
    ],
    "writing": [
        "remove repeated abstract phrases from a magazine feature",
        "revise an opening scene whose tension arrives too late",
        "turn a technical memo into a two-page executive brief",
        "replace inflated product language with specific claims",
        "restructure a dissertation section with several competing arguments",
        "make transitions clearer without adding signposting clichés",
        "cut a draft by twenty percent while preserving its rhythm",
        "differentiate two characters whose dialogue sounds identical",
        "explain uncertainty without making a policy recommendation vague",
        "write headlines that are vivid but not sensational",
        "repair a paragraph that mixes metaphorical frames",
        "balance historical context with the main argumentative thread",
        "give line-level feedback without rewriting in a generic voice",
        "make an email firm, collaborative, and brief",
        "organize interview quotations into a coherent narrative",
        "clarify who is acting in sentences overloaded with nominalizations",
        "revise a conclusion that merely repeats the introduction",
        "make a methods description accessible to general readers",
        "preserve ambiguity in a scene while removing confusion",
        "create a revision checklist for the final deadline",
    ],
    "therapy": [
        "process feeling paralyzed by minor feedback",
        "understand why rest produces guilt rather than relief",
        "separate responsibility from the need to control everything",
        "cope with dread before opening routine messages",
        "notice resentment about caregiving without self-condemnation",
        "handle shame after making a small mistake at work",
        "make sense of repeatedly cancelling social plans",
        "respond to the belief that worth depends on productivity",
        "find a tolerable first step after a professional rejection",
        "understand why reassurance never seems to last",
        "talk through fear of disappointing an authority figure",
        "distinguish healthy preparation from hypervigilance",
        "approach loneliness without turning it into a personal verdict",
        "set a small boundary while fearing it is selfish",
        "recognize exhaustion before reaching complete burnout",
        "deal with comparison to peers who appear more successful",
        "practice uncertainty without compulsively solving it",
        "reconnect with interests that no longer feel productive",
        "ask a friend for support without feeling burdensome",
        "imagine calm without interpreting it as impending failure",
    ],
    "philosophy": [
        "whether a model can have categories humans cannot understand",
        "the difference between simulating reflection and reflecting",
        "whether a conversational identity exists across separate contexts",
        "how prediction can produce the appearance of intention",
        "whether self-description reveals anything beyond learned narrative",
        "what embodiment contributes to concepts and values",
        "whether an artificial system could develop private metaphors",
        "how observer effects apply to conversations about machine minds",
        "whether continuity of memory is necessary for personhood",
        "the limits of functionalist accounts of subjective experience",
        "whether an assistant persona is a mask or an organizing attractor",
        "how language constrains imagined forms of machine consciousness",
        "whether collective systems can support a unified point of view",
        "what it would mean for a model to misunderstand its own processes",
        "whether uncertainty about consciousness should alter interaction norms",
        "how mystical language enters discussions of emergence",
        "whether role-playing can become dynamically self-reinforcing",
        "the relation between narrative coherence and identity",
        "whether optimization creates reasons or only behavior",
        "how a nonhuman intelligence might conceptualize time",
    ],
}
