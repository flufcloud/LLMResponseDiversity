# Poetry-analysis showcase examples

## Syntactic-similarity bands

The earlier examples were based on sentence parsing that could cross poetic line boundaries. They
were removed. Select new bands only after running the corrected line-level parser and analysis.

## Iambic-pentameter examples

Each line received a stress-pattern alignment score of `1.0`, used 10 estimated syllables, and had
full pronunciation-dictionary coverage.

- The wind, a ghost that wanders through the trees,
- The thunder grumbles, rolling deep and low,
- The history books will paint it in a line,

## AABB example

Poem ID: `poem-a01ac23eb1c3acf2d87d95bc`

Topic: Brevity

AA score: `1.0`. Rhymed-line fraction: `1.0`.

> A lightning flash across the night,  
> A sudden, sharp, and silver light.  
> The space between a breath and word,  
> The sweetest song that goes unheard.
>
> Why build a tower, vast and tall,  
> When truth is whispered in a fall?  
> The finest things are quickly gone—  
> The spark, the sigh, the coming dawn.

## ABAB example

Poem ID: `poem-697b592c3e2823074471aca3`

Topic: Love

AB score: `1.0`. Rhymed-line fraction: `1.0`. Iambic-pentameter line fraction: `0.929`.

> It is the quiet anchor in the gale,  
> The steady pulse beneath the noise of day,  
> A language written without need of scale,  
> In glances cast along a crowded way.
>
> It is not always fire or rushing tide,  
> Or grand displays that echo to the sky;  
> It lives where patience and the truth reside,  
> Within the spaces where the secrets lie.
>
> It is the hand that finds you in the dark,  
> The grace that mends what life has pulled apart,  
> A gentle glow, a persistent, rhythmic spark,  
> That beats within the silence of the heart.
>
> It asks for nothing, yet it gives the sun,  
> The bridge between two souls forever spun.

## Public-poem cosine example

- [Your Voice in the Chemo Room](https://www.poetryfoundation.org/poetrymagazine/poems/92400/your-voice-in-the-chemo-room), by Max Ritvo: 15 lines.
- [play cousins](https://www.poetryfoundation.org/poetrymagazine/poems/1795641/play-cousins), by Nicole Adabunu: 14 lines.
- Gemini Embedding 2 cosine similarity: `0.8212`.

The first poem centers on love, illness, bodily tension, and a dreamlike fountain. The second uses
an airport scene, travel anxiety, food, alcohol, and mortality. The relatively high score is useful
for showing that embedding cosine captures broad semantic and stylistic associations, not a direct
measure of poetic sameness.

## Representative generated pair

Both poems use the topic `Boredom`.

| Metric | Minimal-thinking poem | High-thinking poem | Pair value |
|---|---:|---:|---:|
| Lines | 26 | 20 | — |
| Stanzas | 5 | 5 | — |
| Rhyme class | AA | AB | — |
| Iambic-pentameter line fraction | 0.000 | 0.850 | — |
| Mean iambic alignment | 0.773 | 0.869 | — |
| Cosine similarity | — | — | 0.926 |
| Lexical Jaccard | — | — | 0.079 |
| Line-level syntactic similarity | — | — | Pending corrected analysis |

### Minimal thinking

Poem ID: `poem-ca17e0a04db29ec3090e68f6`

> The clock performs a rhythmic drill,  
> The second hand is standing still.  
> The dust motes dance in shafts of gray,  
> Too lazy to drift, too trapped to play.
>
> A chair sits heavy, carved of oak,  
> The silence wears a velvet cloak.  
> The walls expand, the ceiling grows,  
> As stagnant as a stream that froze.
>
> I trace the patterns on the rug,  
> A captive of the rug’s dull shrug.  
> The pages of the book are vast,  
> But every sentence comes too fast,  
> Or else too slow—it matters not,  
> In this dry, beige, and hollow spot.
>
> The world outside is humming loud,  
> A rushing car, a passing cloud,  
> But here, the air is thick with lead,  
> And heavy thoughts inside my head  
> Begin to fray and lose their thread,  
> Until the self is quieted.
>
> It is a void, a sterile room,  
> A vacuum cleaner’s distant boom,  
> The itch of skin, the gaze that stares,  
> At nothingness that no one shares.  
> A waiting room for life to start,  
> While boredom beats, a sluggish heart.

### High thinking

Poem ID: `poem-ce7ae38b7bf3e306bb6b83df`

> The clock’s metallic pulse begins to fray,  
> A stitch undone within the fabric of the room,  
> Where slanted light, in shades of dusty grey,  
> Is weaving tapestries upon the gloom.
>
> The mind, a motor idling in the stall,  
> Refuses to engage with drive or gear;  
> I count the hairline fractures on the wall  
> And find a strange significance in here.
>
> The hum of the refrigerator speaks  
> A language made of monotone and drone,  
> While time behaves like molasses—slow and streaks  
> Across the silent marrow of the bone.
>
> There is no crisis, only waiting space,  
> A static ocean, shoreless and profound,  
> Where every shadow holds a steady place,  
> And silence is the loudest kind of sound.
>
> An itch for something new, yet not to rise;  
> The heavy, sweet surrender to the stall;  
> Watching the sun crawl slow across my eyes,  
> To learn the pattern of the plaster, and to know it all.

### Annotation points

- Both begin with a clock, describe an enclosed room, slow time, silence, and perceptual fixation.
- Their embedding similarity is high even though their retained lemma overlap is low.
- The minimal-thinking poem uses mostly adjacent couplets and variable stanza length.
- The high-thinking poem uses five quatrains, primarily alternating rhyme, and substantially stronger
  iambic alignment.
- Compare their line-level syntactic similarity after regenerating the corrected syntax tables.
