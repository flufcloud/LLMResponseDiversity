# Lexical-addition analysis

This analysis measures how many retained content lemmas each successive poem adds to the set seen
for its topic. It uses the same part-of-speech filter, stop-word removal, explicit-negation removal,
and lemmatization as the existing poetry lexical analysis.

For the basic corpus, every topic contributes its existing 20 poems. Each repetition randomly
orders those poems.

For the adjective corpus, every topic contributes a balanced 20-poem sample. All 15 adjectives
contribute one poem, five randomly selected adjectives contribute a second poem, and the resulting
20 poems are randomly ordered. The extra allocations rotate across 1,000 repetitions.

The adjective stage reuses the row-level lemmas from the adjective similarity pipeline when they
are available. The basic corpus is tagged separately because its earlier lexical outputs do not
retain one lemma list per poem.

The outputs report cumulative unique lemmas, new lemmas added by the incoming poem, the fraction of
the incoming poem's distinct lemmas that are new, and the probability that the incoming poem adds
no new lemma.
